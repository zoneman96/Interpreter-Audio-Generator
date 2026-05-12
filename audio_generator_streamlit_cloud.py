#!/usr/bin/env python3
"""
Cloud-safe Streamlit dashboard for interpreter audio generation and practice.

Run locally:
    streamlit run audio_generator_streamlit_cloud.py

Deploy with Streamlit Community Cloud using this file as the entrypoint.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import io
import json
import os
import shutil
import tempfile
import csv
import time
import base64
import hashlib
import re
import uuid
import html
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

APP_DIR = Path(__file__).resolve().parent
CLOUD_MODE = True
RUNTIME_DIR = Path(tempfile.gettempdir()) / "interpreter_audio_generator_web"
RUNTIME_OUTPUT_DIR = RUNTIME_DIR / "audio_output"
RUNTIME_CACHE_DIR = RUNTIME_DIR / "cache"
DEFAULT_GENERATOR_PATH = APP_DIR / "Current_Audio_Generator_Script_JSON_to_MP3_V7.py"
DEFAULT_STUDY_PROGRESS_DIR = RUNTIME_DIR / "study_progress"
DEFAULT_PRACTICE_LOG_PATH = DEFAULT_STUDY_PROGRESS_DIR / "practice_log.csv"
DEFAULT_TERM_REVIEW_LOG_PATH = DEFAULT_STUDY_PROGRESS_DIR / "term_review_log.csv"
DEFAULT_TERM_SRS_STATE_PATH = DEFAULT_STUDY_PROGRESS_DIR / "term_srs_state.csv"
DEFAULT_SIMULTANEOUS_LOG_PATH = DEFAULT_STUDY_PROGRESS_DIR / "simultaneous_practice_log.csv"

APPROVED_EDGE_VOICES = {
    "en-US-AvaNeural", "en-US-AndrewNeural", "en-US-EmmaNeural", "en-US-BrianNeural",
    "en-US-AnaNeural", "en-US-AndrewMultilingualNeural", "en-US-AriaNeural",
    "en-US-AvaMultilingualNeural", "en-US-BrianMultilingualNeural", "en-US-ChristopherNeural",
    "en-US-EmmaMultilingualNeural", "en-US-EricNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-US-MichelleNeural", "en-US-RogerNeural", "en-US-SteffanNeural",
    "es-ES-XimenaNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-ES-AlvaroNeural",
    "es-ES-ElviraNeural", "es-US-AlonsoNeural", "es-US-PalomaNeural",
}

REQUIRED_TOP_LEVEL_FIELDS = [
    "script_id", "category", "subcategory", "title", "format", "terms_used",
    "english_script", "spanish_script", "paired_segments",
]


def load_generator_module(generator_path: Path):
    if not generator_path.exists():
        raise FileNotFoundError(f"Generator script not found: {generator_path}")
    spec = importlib.util.spec_from_file_location("current_audio_generator_v7", generator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generator module from {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_script_id(data: dict[str, Any], fallback: str) -> str:
    return str(data.get("script_id") or fallback).strip() or fallback


def load_json_bytes(name: str, raw: bytes) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
        if not isinstance(payload, dict):
            return None, "Top-level JSON value is not an object."
        return payload, None
    except Exception as exc:
        return None, f"Could not parse {name}: {exc}"



def make_zip_bytes(paths: list[Path], base_dir: Path | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        seen: set[str] = set()
        for path in paths:
            if not path.exists() or not path.is_file():
                continue
            if base_dir is not None:
                try:
                    arcname = str(path.relative_to(base_dir))
                except ValueError:
                    arcname = path.name
            else:
                arcname = path.name
            # Avoid duplicate archive names.
            original = arcname
            counter = 2
            while arcname in seen:
                p = Path(original)
                arcname = str(p.with_name(f"{p.stem}_{counter}{p.suffix}"))
                counter += 1
            seen.add(arcname)
            zf.write(path, arcname)
    buffer.seek(0)
    return buffer.getvalue()

def collect_voice_names(audio_profile: dict[str, Any]) -> list[str]:
    voices: list[str] = []
    defaults = audio_profile.get("default_voices", {}) or {}
    if isinstance(defaults, dict):
        voices.extend(str(v).strip() for v in defaults.values() if str(v).strip())

    assignments = audio_profile.get("voice_assignments", {}) or {}
    if isinstance(assignments, dict):
        for value in assignments.values():
            if isinstance(value, dict):
                voices.extend(str(v).strip() for key, v in value.items() if key in {"english", "spanish"} and str(v).strip())
    return sorted(set(voices))


def validate_json_payload(data: dict[str, Any]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []

    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in data:
            warnings.append(f"Missing recommended top-level field: {field}")

    terms = data.get("terms_used", []) or []
    if not isinstance(terms, list):
        warnings.append("terms_used should be a list.")

    paired = data.get("paired_segments", []) or []
    if not isinstance(paired, list) or not paired:
        warnings.append("No paired_segments found. The generator may fall back to paragraph splitting.")
        return warnings, errors

    for i, seg in enumerate(paired, start=1):
        if not isinstance(seg, dict):
            errors.append(f"Segment {i}: segment is not a JSON object.")
            continue
        if not str(seg.get("english", "")).strip():
            errors.append(f"Segment {i}: missing English text.")
        if not str(seg.get("spanish", "")).strip():
            errors.append(f"Segment {i}: missing Spanish text.")

        order = seg.get("playback_order")
        if order not in (["english", "spanish"], ["spanish", "english"]):
            warnings.append(f"Segment {i}: playback_order should be ['english','spanish'] or ['spanish','english'].")

        source_language = str(seg.get("source_language") or seg.get("original_language") or "").strip().lower()
        if source_language in {"english", "spanish"} and isinstance(order, list) and order:
            if source_language != str(order[0]).lower():
                warnings.append(
                    f"Segment {i}: source_language is {source_language!r}, but playback_order starts with {order[0]!r}."
                )
        elif not source_language:
            warnings.append(f"Segment {i}: missing source_language/original_language.")

        if not str(seg.get("speaker", "")).strip():
            warnings.append(f"Segment {i}: missing speaker label; speaker-specific voices may not apply.")

    audio_profile = data.get("audio_profile", {}) or {}
    if isinstance(audio_profile, dict) and audio_profile:
        voices = collect_voice_names(audio_profile)
        unapproved = [v for v in voices if v not in APPROVED_EDGE_VOICES]
        if unapproved:
            warnings.append("Unapproved or unconfirmed Edge TTS voice(s): " + ", ".join(unapproved))

        assignments = audio_profile.get("voice_assignments", {}) or {}
        if isinstance(assignments, dict):
            assigned = {str(k).strip() for k in assignments.keys()}
            speakers = {str(seg.get("speaker", "")).strip() for seg in paired if isinstance(seg, dict) and str(seg.get("speaker", "")).strip()}
            missing = sorted(speakers - assigned)
            if missing:
                warnings.append("Speakers without explicit audio_profile assignment: " + ", ".join(missing[:12]))

    return warnings, errors


def summarize_payload(data: dict[str, Any], filename: str) -> dict[str, Any]:
    paired = data.get("paired_segments", []) or []
    if not isinstance(paired, list):
        paired = []
    terms = data.get("terms_used", []) or []
    if not isinstance(terms, list):
        terms = []
    english_source = 0
    spanish_source = 0
    speakers: set[str] = set()
    segment_types: set[str] = set()
    for seg in paired:
        if not isinstance(seg, dict):
            continue
        src = str(seg.get("source_language") or seg.get("original_language") or "").strip().lower()
        if src == "english":
            english_source += 1
        elif src == "spanish":
            spanish_source += 1
        if str(seg.get("speaker", "")).strip():
            speakers.add(str(seg.get("speaker", "")).strip())
        if str(seg.get("segment_type", "")).strip():
            segment_types.add(str(seg.get("segment_type", "")).strip())
    return {
        "file": filename,
        "script_id": safe_script_id(data, Path(filename).stem),
        "title": data.get("title", ""),
        "format": data.get("format", ""),
        "segments": len(paired),
        "terms": len(terms),
        "english_source": english_source,
        "spanish_source": spanish_source,
        "speakers": ", ".join(sorted(speakers))[:180],
        "segment_types": ", ".join(sorted(segment_types))[:180],
    }


async def run_generator(generator_path: Path, json_files: list[Path], output_dir: Path, settings: dict[str, Any]) -> list[Path]:
    generator = load_generator_module(generator_path)

    generator.OUTPUT_DIR = output_dir
    generator.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for key, value in settings.items():
        if hasattr(generator, key):
            setattr(generator, key, value)

    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg was not found on PATH. Install ffmpeg before generating audio.")

    await generator.initialize_voice_validation()

    outputs: list[Path] = []
    for path in json_files:
        results = await generator.build_audio_for_script(path)
        outputs.extend(results or [])

    if getattr(generator, "GENERATE_COMBINED_AUDIO_ALL_SCRIPTS", False):
        all_script_mp3s = (
            generator.get_finished_script_rendition_mp3s(output_dir)
            if getattr(generator, "SPLIT_FLASHCARDS_AND_SCRIPT", False)
            else generator.get_all_finished_script_mp3s(output_dir)
        )
        if all_script_mp3s:
            combined_out = output_dir / "combined_audio_all_scripts.mp3"
            generator.concat_mp3s_with_gap(
                all_script_mp3s,
                combined_out,
                gap_seconds=getattr(generator, "COMBINED_SCRIPT_GAP", 3.0),
            )
            outputs.append(combined_out)
    return outputs





# -----------------------------
# Interactive practice audio helpers
# -----------------------------
DEFAULT_PRACTICE_VOICES = {
    "english": "en-US-JennyNeural",
    "spanish": "es-MX-DaliaNeural",
}


def sanitize_filename_part(value: str, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._") or "item"
    return cleaned[:max_len]


def short_hash(*parts: str) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:12]


def get_rate_for_language(speed_mode: str, lang: str) -> str:
    presets = {
        "learning": {"english": "-17%", "spanish": "-17%"},
        "normal": {"english": "-4%", "spanish": "-4%"},
        "fast": {"english": "+6%", "spanish": "+6%"},
    }
    return presets.get(speed_mode, presets["learning"]).get(lang, "-17%")


def get_practice_voice(data: dict[str, Any], seg: dict[str, Any], lang: str, use_audio_profile: bool = True) -> str:
    """Resolve a voice for the practice player using the same intended fallback order as the generator."""
    lang = str(lang).strip().lower()
    fallback = DEFAULT_PRACTICE_VOICES.get(lang, "en-US-JennyNeural")
    if not use_audio_profile:
        return fallback

    def voice_from_entry(entry: Any) -> str:
        if isinstance(entry, str):
            return entry.strip()
        if isinstance(entry, dict):
            return str(entry.get(lang) or entry.get("voice") or entry.get("default") or "").strip()
        return ""

    # Segment-level override.
    if seg.get("voice"):
        voice = voice_from_entry(seg.get("voice"))
        if voice:
            return voice
    for key in ("voices", "voice_overrides"):
        entry = seg.get(key)
        if isinstance(entry, dict):
            voice = str(entry.get(lang) or "").strip()
            if voice:
                return voice

    profile = data.get("audio_profile") or {}
    if not isinstance(profile, dict):
        return fallback

    assignments = profile.get("voice_assignments") or {}
    speaker = str(seg.get("speaker") or "").strip().upper()
    if isinstance(assignments, dict) and speaker:
        # Match exact key first, then upper-case normalized key.
        entry = assignments.get(seg.get("speaker")) or assignments.get(speaker)
        if entry is None:
            for k, v in assignments.items():
                if str(k).strip().upper() == speaker:
                    entry = v
                    break
        voice = voice_from_entry(entry)
        if voice:
            return voice

    defaults = profile.get("default_voices") or {}
    if isinstance(defaults, dict):
        voice = str(defaults.get(lang) or "").strip()
        if voice:
            return voice
    return fallback


def get_practice_voice_for_chunk(data: dict[str, Any], chunk: dict[str, Any], lang: str, use_audio_profile: bool = True) -> str:
    """Resolve a practice voice for a simultaneous chunk.

    Single-chunk mode uses the segment's speaker voice. Full-passage mode may wrap
    several paired_segments; when they share a speaker, keep that assigned speaker
    voice instead of reverting to the global default. If there are multiple
    speakers, use the first segment's assigned voice as the practical fallback,
    because one continuous generated MP3 can only use one voice.
    """
    fallback = DEFAULT_PRACTICE_VOICES.get(str(lang).strip().lower(), "en-US-JennyNeural")
    chunk_segments = chunk.get("segments") or []
    if isinstance(chunk_segments, list) and chunk_segments:
        usable_segments = [seg for seg in chunk_segments if isinstance(seg, dict)]
        if usable_segments:
            return get_practice_voice(data, usable_segments[0], lang, use_audio_profile)

    # Paragraph/thought chunks keep a direct speaker field on the chunk.
    pseudo_segment = {"speaker": chunk.get("speaker", "")}
    return get_practice_voice(data, pseudo_segment, lang, use_audio_profile) if use_audio_profile else fallback


async def generate_practice_audio_file(text: str, voice: str, rate: str, out_path: Path, generator_path: Path | None = None) -> None:
    """Generate a small MP3 for the interactive practice player."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    # Prefer the generator's hardened tts_to_mp3 when available, because it includes
    # voice validation and fallback behavior.
    if generator_path and generator_path.exists():
        try:
            generator = load_generator_module(generator_path)
            if hasattr(generator, "VERIFY_TTS_VOICES"):
                generator.VERIFY_TTS_VOICES = True
            if hasattr(generator, "initialize_voice_validation"):
                await generator.initialize_voice_validation()
            if hasattr(generator, "tts_to_mp3"):
                await generator.tts_to_mp3(text, voice, rate, out_path)
                return
        except Exception:
            # Fall back to direct edge_tts below so one helper mismatch does not break practice mode.
            pass

    import edge_tts
    communicate = edge_tts.Communicate(text=str(text), voice=voice, rate=rate)
    await communicate.save(str(out_path))


def get_practice_audio_path(
    cache_dir: Path,
    script_id: str,
    seg_index: Any,
    role: str,
    lang: str,
    text: str,
    voice: str,
    rate: str,
) -> Path:
    key = short_hash(script_id, str(seg_index), role, lang, text, voice, rate)
    filename = f"{sanitize_filename_part(script_id)}_seg_{int(seg_index):04d}_{role}_{lang}_{key}.mp3" if str(seg_index).isdigit() else f"{sanitize_filename_part(script_id)}_{role}_{lang}_{key}.mp3"
    return cache_dir / filename


def render_audio_player(
    path: Path,
    label: str,
    autoplay: bool = False,
    show_native_player: bool = True,
    nonce: str = "",
    playback_speed: float = 1.0,
) -> None:
    """Render an MP3 with a forced-refresh HTML audio element.

    Uses components.html instead of st.markdown so long base64 data URLs are
    rendered as an audio element rather than occasionally appearing as raw HTML
    text in Streamlit. This renderer is used by the Consecutive and Term Review
    tabs; the simultaneous tab has a speed-aware variant below.
    """
    if not path.exists():
        st.warning(f"Audio file not found: {path.name}")
        return

    audio_bytes = path.read_bytes()
    if not audio_bytes:
        st.warning(f"Audio file is empty: {path.name}")
        return

    content_key = short_hash(path.name, str(path.stat().st_mtime_ns), str(len(audio_bytes)), nonce, str(playback_speed))
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    element_id = f"practice_audio_{sanitize_filename_part(content_key)}"
    autoplay_attr = "autoplay" if autoplay else ""
    controls_attr = "controls" if show_native_player else ""
    audio_src = f"data:audio/mpeg;base64,{audio_b64}"
    safe_speed = max(0.5, min(2.0, float(playback_speed or 1.0)))

    component_html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ margin: 0; padding: 0; background: transparent; }}
          audio {{ width: 100%; display: block; }}
          .caption {{
            margin-top: 4px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 12px;
            color: #6b7280;
          }}
        </style>
      </head>
      <body>
        <audio id="{element_id}" {controls_attr} {autoplay_attr} preload="auto" src="{audio_src}"></audio>
        <div class="caption">{html.escape(label)} · {safe_speed:.2f}x · audio key: {content_key}</div>
        <script>
          const el = document.getElementById("{element_id}");
          if (el) {{
            el.playbackRate = {safe_speed};
            el.defaultPlaybackRate = {safe_speed};
          }}
        </script>
      </body>
    </html>
    """
    components.html(component_html, height=76, scrolling=False)


def render_simultaneous_audio_player(
    path: Path,
    label: str,
    autoplay: bool = False,
    show_native_player: bool = True,
    nonce: str = "",
    playback_speed: float = 1.0,
    loop: bool = False,
) -> None:
    """Render simultaneous-practice audio in an iframe-safe HTML component.

    Phase 2D.1 originally used st.markdown(unsafe_allow_html=True) for this
    custom audio tag. On some Streamlit/browser combinations, the long base64
    data URL can be escaped and shown as raw HTML text. Using components.html
    renders the audio element directly and avoids that raw HTML display issue
    while preserving playback speed and repeat-loop support.
    """
    if not path.exists():
        st.warning(f"Audio file not found: {path.name}")
        return

    audio_bytes = path.read_bytes()
    if not audio_bytes:
        st.warning(f"Audio file is empty: {path.name}")
        return

    content_key = short_hash(
        path.name,
        str(path.stat().st_mtime_ns),
        str(len(audio_bytes)),
        nonce,
        str(playback_speed),
        str(loop),
    )
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    element_id = f"sim_audio_{sanitize_filename_part(content_key)}"
    autoplay_attr = "autoplay" if autoplay else ""
    controls_attr = "controls" if show_native_player else ""
    loop_attr = "loop" if loop else ""
    safe_speed = max(0.5, min(2.0, float(playback_speed or 1.0)))
    audio_src = f"data:audio/mpeg;base64,{audio_b64}"

    component_html = f"""
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ margin: 0; padding: 0; background: transparent; }}
          audio {{ width: 100%; display: block; }}
          .caption {{
            margin-top: 4px;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            font-size: 12px;
            color: #6b7280;
          }}
        </style>
      </head>
      <body>
        <audio id="{element_id}" {controls_attr} {autoplay_attr} {loop_attr} preload="auto" src="{audio_src}"></audio>
        <div class="caption">{html.escape(label)} · {safe_speed:.2f}x{(' · loop on' if loop else '')} · audio key: {content_key}</div>
        <script>
          const el = document.getElementById("{element_id}");
          if (el) {{
            el.playbackRate = {safe_speed};
            el.defaultPlaybackRate = {safe_speed};
          }}
        </script>
      </body>
    </html>
    """
    components.html(component_html, height=76, scrolling=False)


# Phase 2E UI helpers -------------------------------------------------------
def render_text_card(title: str, text: str, meta: str = "", tone: str = "source") -> None:
    """Render a clean text card for practice source/reference text."""
    tone_class = "reference" if tone == "reference" else "source"
    safe_title = html.escape(str(title or ""))
    safe_meta = html.escape(str(meta or ""))
    safe_text = html.escape(str(text or "")).replace("\n", "<br>")
    meta_html = f'<div class="practice-card-meta">{safe_meta}</div>' if safe_meta else ''
    st.markdown(
        f"""
        <div class="practice-card {tone_class}">
          <div class="practice-card-title">{safe_title}</div>
          {meta_html}
          <div class="practice-card-body">{safe_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hidden_reference_card(title: str = "Reference hidden", prompt: str = "Reveal when ready.") -> None:
    st.markdown(
        f"""
        <div class="practice-card hidden-reference">
          <div class="practice-card-title">{html.escape(title)}</div>
          <div class="practice-card-body muted">{html.escape(prompt)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_side_by_side_text(
    source_text: str,
    target_text: str | None,
    source_lang: str,
    target_lang: str,
    speaker: str = "",
    segment_type: str = "",
    revealed: bool = False,
    source_visible: bool = True,
    context_label: str = "Reference interpretation",
) -> None:
    """Responsive source/reference workspace for Consecutive and Simultaneous practice."""
    left, right = st.columns(2, gap="large")
    meta_bits = [b for b in [f"Speaker: {speaker}" if speaker else "", f"Type: {segment_type}" if segment_type else ""] if b]
    meta = " · ".join(meta_bits)
    with left:
        if source_visible:
            render_text_card(f"Source — {str(source_lang).title()}", source_text, meta=meta, tone="source")
        else:
            render_hidden_reference_card("Source transcript hidden", "Audio-only mode is on. Interpret aloud while listening.")
    with right:
        if revealed and target_text:
            render_text_card(f"{context_label} — {str(target_lang).title()}", target_text, meta=meta, tone="reference")
        else:
            render_hidden_reference_card("Reference hidden", "Reveal the reference when you are ready to compare.")



def format_lang_label(lang: str) -> str:
    lang = str(lang or "").strip().lower()
    if lang == "english":
        return "EN"
    if lang == "spanish":
        return "ES"
    return lang[:2].upper() if lang else "—"


def render_compact_segment_meta(
    primary_label: str,
    source_lang: str,
    target_lang: str,
    word_count: int | None = None,
    speaker: str = "",
    segment_type: str = "",
    extra: str = "",
) -> None:
    """Render small, single-line metadata instead of large metric cards."""
    parts = [f"{format_lang_label(source_lang)} → {format_lang_label(target_lang)}"]
    if word_count is not None:
        parts.append(f"{int(word_count)} words")
    if speaker:
        parts.append(str(speaker))
    if segment_type:
        parts.append(str(segment_type))
    if extra:
        parts.append(str(extra))

    safe_primary = html.escape(str(primary_label or ""))
    safe_parts = [html.escape(str(part)) for part in parts if str(part).strip()]
    pills = "".join(f'<span class="compact-meta-pill">{part}</span>' for part in safe_parts)
    st.markdown(
        f"""
        <div class="compact-segment-meta">
          <span class="compact-meta-primary">{safe_primary}</span>
          <span class="compact-meta-pills">{pills}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def inject_phase2e_css() -> None:
    st.markdown(
        """
        <style>
          .practice-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            margin: 0.35rem 0 0.85rem 0;
            background: rgba(255, 255, 255, 0.72);
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .practice-card.reference { border-left: 5px solid rgba(46, 125, 50, 0.55); }
          .practice-card.source { border-left: 5px solid rgba(30, 136, 229, 0.55); }
          .practice-card.hidden-reference {
            border-style: dashed;
            background: rgba(250, 250, 250, 0.62);
          }
          .practice-card-title {
            font-weight: 700;
            font-size: 0.92rem;
            text-transform: uppercase;
            letter-spacing: 0.035em;
            margin-bottom: 0.35rem;
          }
          .practice-card-meta {
            font-size: 0.82rem;
            color: rgba(49, 51, 63, 0.68);
            margin-bottom: 0.55rem;
          }
          .practice-card-body { font-size: 1rem; line-height: 1.55; }
          .practice-card-body.muted {
            color: rgba(49, 51, 63, 0.58);
            font-style: italic;
          }
          .compact-segment-meta {
            display: flex;
            align-items: center;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin: 0.15rem 0 0.75rem 0;
            padding: 0.35rem 0.45rem;
            border-radius: 999px;
            background: rgba(250, 250, 250, 0.72);
            border: 1px solid rgba(49, 51, 63, 0.10);
            font-size: 0.78rem;
            line-height: 1.2;
          }
          .compact-meta-primary {
            font-weight: 700;
            color: rgba(49, 51, 63, 0.78);
            margin-right: 0.1rem;
          }
          .compact-meta-pills {
            display: inline-flex;
            gap: 0.28rem;
            flex-wrap: wrap;
          }
          .compact-meta-pill {
            display: inline-block;
            padding: 0.14rem 0.42rem;
            border-radius: 999px;
            background: rgba(49, 51, 63, 0.055);
            color: rgba(49, 51, 63, 0.72);
            max-width: 18rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
          @media (max-width: 760px) {
            .practice-card { padding: 0.9rem; }
            .practice-card-body { font-size: 0.96rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

ERROR_TAGS = [
    "Omission",
    "Number/date/time error",
    "Wrong legal term",
    "Wrong speaker/actor",
    "Register/style issue",
    "Sequence/chronology error",
    "Source-language interference",
    "Memory/retention breakdown",
]


def get_segment_source_and_target(seg: dict[str, Any], direction: str = "Source language first") -> tuple[str, str]:
    """Return (source_lang, target_lang) for a paired segment."""
    if direction == "English → Spanish only":
        return "english", "spanish"
    if direction == "Spanish → English only":
        return "spanish", "english"

    order = seg.get("playback_order")
    if isinstance(order, list) and order and str(order[0]).lower() in {"english", "spanish"}:
        source = str(order[0]).lower()
    else:
        source = str(seg.get("source_language") or seg.get("original_language") or "english").lower()
        if source not in {"english", "spanish"}:
            source = "english"
    target = "spanish" if source == "english" else "english"
    return source, target


def estimate_dynamic_pause_seconds(text: str, ratio: float, min_pause: float, max_pause: float) -> float:
    """Estimate practice pause without creating audio. Audio generator uses actual MP3 duration."""
    words = max(1, len(str(text).split()))
    # Conservative speech estimate: about 145 words/minute.
    estimated_audio_seconds = words / 145 * 60
    return round(max(min_pause, min(max_pause, estimated_audio_seconds * ratio)), 1)


def filter_practice_segments(data: dict[str, Any], direction: str, segment_type_filter: list[str], speaker_filter: list[str]) -> list[dict[str, Any]]:
    paired = data.get("paired_segments", []) or []
    if not isinstance(paired, list):
        return []
    filtered: list[dict[str, Any]] = []
    for idx, seg in enumerate(paired, start=1):
        if not isinstance(seg, dict):
            continue
        source, target = get_segment_source_and_target(seg, direction)
        if not str(seg.get(source, "")).strip() or not str(seg.get(target, "")).strip():
            continue
        if segment_type_filter and str(seg.get("segment_type", "")).strip() not in segment_type_filter:
            continue
        if speaker_filter and str(seg.get("speaker", "")).strip() not in speaker_filter:
            continue
        copy = dict(seg)
        copy["_segment_index"] = idx
        copy["_source_language"] = source
        copy["_target_language"] = target
        filtered.append(copy)
    return filtered


PRACTICE_LOG_FIELDS = [
    "timestamp", "session_id", "script_id", "title", "practice_direction", "practice_set_size",
    "segment_index", "speaker", "segment_type", "source_language", "target_language",
    "score", "error_tags", "notes", "source_text", "target_text",
]


def practice_log_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=PRACTICE_LOG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in PRACTICE_LOG_FIELDS})
    return output.getvalue()


def normalize_log_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {k: row.get(k, "") for k in PRACTICE_LOG_FIELDS}
    normalized["score"] = str(normalized.get("score", ""))
    return normalized


def load_persistent_practice_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [normalize_log_row(row) for row in csv.DictReader(f)]
    except Exception as exc:
        st.warning(f"Could not read persistent practice log: {exc}")
        return []


def append_persistent_practice_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRACTICE_LOG_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PRACTICE_LOG_FIELDS})


def overwrite_persistent_practice_log(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PRACTICE_LOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in PRACTICE_LOG_FIELDS})


def parse_score(value: Any) -> int | None:
    try:
        score = int(value)
        return score if score in {0, 1, 2, 3} else None
    except Exception:
        return None


def rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=PRACTICE_LOG_FIELDS)
    df = pd.DataFrame([normalize_log_row(r) for r in rows])
    for col in PRACTICE_LOG_FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["score_numeric"] = pd.to_numeric(df["score"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp_dt"].dt.date.astype("string")
    df["direction"] = df["source_language"].str.title() + " → " + df["target_language"].str.title()
    return df


def split_error_tags(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = str(row.get("error_tags", ""))
        for tag in [t.strip() for t in raw.split(";") if t.strip()]:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))




# -----------------------------
# Phase 2C term review helpers
# -----------------------------
TERM_REVIEW_FIELDS = [
    "timestamp", "session_id", "script_id", "title", "source_file",
    "prompt_language", "answer_language", "english", "spanish", "alternatives",
    "score_label", "score_numeric", "notes",
]

TERM_SCORE_OPTIONS = {
    "Easy": 3,
    "Good": 2,
    "Hard": 1,
    "Missed": 0,
}


def normalize_term_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def extract_terms_from_payload(data: dict[str, Any], filename: str) -> list[dict[str, Any]]:
    terms = data.get("terms_used", []) or []
    if not isinstance(terms, list):
        return []
    script_id = safe_script_id(data, Path(filename).stem)
    title = str(data.get("title", ""))
    extracted: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for i, term in enumerate(terms, start=1):
        if not isinstance(term, dict):
            continue
        english = normalize_term_text(term.get("english") or term.get("term") or term.get("source") or "")
        spanish = normalize_term_text(
            term.get("spanish_contextual_choice")
            or term.get("preferred_spanish")
            or term.get("spanish")
            or term.get("target")
            or ""
        )
        alternatives_raw = term.get("spanish_alternatives_from_sheet") or term.get("spanish_alternatives") or []
        if isinstance(alternatives_raw, list):
            alternatives = "; ".join(normalize_term_text(v) for v in alternatives_raw if normalize_term_text(v))
        else:
            alternatives = normalize_term_text(alternatives_raw)
        if not english or not spanish:
            continue
        key = (english.lower(), spanish.lower())
        if key in seen:
            continue
        seen.add(key)
        extracted.append({
            "term_id": f"{script_id}:{i}:{short_hash(english, spanish)}",
            "script_id": script_id,
            "title": title,
            "source_file": filename,
            "english": english,
            "spanish": spanish,
            "alternatives": alternatives,
        })
    return extracted


def load_term_payloads_from_files(files: list[tuple[str, bytes]]) -> tuple[list[tuple[str, dict[str, Any], bytes]], list[str]]:
    payloads: list[tuple[str, dict[str, Any], bytes]] = []
    errors: list[str] = []
    for name, raw in files:
        data, err = load_json_bytes(name, raw)
        if err:
            errors.append(err)
        elif data is not None:
            payloads.append((name, data, raw))
    return payloads, errors


def term_review_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TERM_REVIEW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in TERM_REVIEW_FIELDS})
    return output.getvalue()


def normalize_term_review_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {k: row.get(k, "") for k in TERM_REVIEW_FIELDS}
    normalized["score_numeric"] = str(normalized.get("score_numeric", ""))
    return normalized


def load_term_review_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [normalize_term_review_row(row) for row in csv.DictReader(f)]
    except Exception as exc:
        st.warning(f"Could not read term review log: {exc}")
        return []


def append_term_review_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERM_REVIEW_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in TERM_REVIEW_FIELDS})


def term_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TERM_REVIEW_FIELDS)
    df = pd.DataFrame([normalize_term_review_row(r) for r in rows])
    for col in TERM_REVIEW_FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["score_numeric"] = pd.to_numeric(df["score_numeric"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp_dt"].dt.date.astype("string")
    df["term_key"] = df["english"].str.lower().str.strip() + "||" + df["spanish"].str.lower().str.strip()
    return df


def get_term_stats(term_history_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = term_rows_to_dataframe(term_history_rows)
    if df.empty:
        return pd.DataFrame(columns=["english", "spanish", "attempts", "average_score", "misses", "last_seen"])
    grouped = (
        df.dropna(subset=["score_numeric"])
        .groupby(["english", "spanish"], as_index=False)
        .agg(
            attempts=("score_numeric", "count"),
            average_score=("score_numeric", "mean"),
            misses=("score_numeric", lambda s: int((s <= 1).sum())),
            hard_or_missed=("score_numeric", lambda s: int((s <= 1).sum())),
            last_seen=("timestamp_dt", "max"),
        )
    )
    return grouped


def rank_terms_for_review(terms: list[dict[str, Any]], term_history_rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if not terms:
        return []
    stats = get_term_stats(term_history_rows)
    stats_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if not stats.empty:
        for row in stats.to_dict("records"):
            stats_by_key[(str(row.get("english", "")).lower().strip(), str(row.get("spanish", "")).lower().strip())] = row

    ranked: list[dict[str, Any]] = []
    for pos, term in enumerate(terms):
        key = (str(term.get("english", "")).lower().strip(), str(term.get("spanish", "")).lower().strip())
        stat = stats_by_key.get(key, {})
        attempts = int(stat.get("attempts") or 0)
        avg = float(stat.get("average_score") if pd.notna(stat.get("average_score")) else 3.0) if stat else 3.0
        misses = int(stat.get("misses") or 0)
        priority = 0
        if attempts == 0:
            priority += 50
        priority += misses * 25
        priority += max(0, int(round((3 - avg) * 10)))
        priority -= min(attempts, 10)
        item = dict(term)
        item["attempts"] = attempts
        item["average_score"] = avg if attempts else None
        item["misses"] = misses
        item["priority"] = priority
        item["original_order"] = pos
        ranked.append(item)

    if mode == "New terms first":
        ranked.sort(key=lambda t: (t.get("attempts", 0) != 0, t.get("original_order", 0)))
    elif mode == "Hard/missed first":
        ranked.sort(key=lambda t: (-int(t.get("priority", 0)), t.get("original_order", 0)))
    elif mode == "Random":
        # stable random-ish shuffle for Streamlit reruns; Start New Session refreshes the seed
        seed = st.session_state.get("term_review_shuffle_seed", "default")
        ranked.sort(key=lambda t: short_hash(seed, t.get("term_id", "")))
    else:
        ranked.sort(key=lambda t: t.get("original_order", 0))
    return ranked



# -----------------------------
# Phase 2C.2 spaced repetition helpers
# -----------------------------
TERM_SRS_FIELDS = [
    "term_key", "english", "spanish", "script_ids", "titles", "source_files",
    "created_at", "last_reviewed_at", "next_review_at", "last_score_label",
    "last_score_numeric", "attempts", "misses", "lapses", "interval_days", "mastery_level",
]


def term_srs_key(term: dict[str, Any]) -> str:
    """Stable key across scripts for the same English/Spanish pair."""
    return short_hash(
        normalize_term_text(term.get("english", "")).lower(),
        normalize_term_text(term.get("spanish", "")).lower(),
    )


def today_date() -> datetime.date:
    return datetime.now().date()


def parse_date_value(value: Any) -> datetime.date | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except Exception:
        try:
            return datetime.strptime(raw[:10], "%Y-%m-%d").date()
        except Exception:
            return None


def normalize_srs_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {k: row.get(k, "") for k in TERM_SRS_FIELDS}
    for numeric in ["last_score_numeric", "attempts", "misses", "lapses", "interval_days", "mastery_level"]:
        normalized[numeric] = str(normalized.get(numeric, "") or "0")
    return normalized


def load_term_srs_state(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [normalize_srs_row(row) for row in csv.DictReader(f)]
    except Exception as exc:
        st.warning(f"Could not read term SRS state: {exc}")
        return []


def save_term_srs_state(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERM_SRS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in TERM_SRS_FIELDS})


def srs_rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TERM_SRS_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in TERM_SRS_FIELDS})
    return output.getvalue()


def term_srs_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=TERM_SRS_FIELDS)
    df = pd.DataFrame([normalize_srs_row(r) for r in rows])
    for col in TERM_SRS_FIELDS:
        if col not in df.columns:
            df[col] = ""
    for col in ["last_score_numeric", "attempts", "misses", "lapses", "interval_days", "mastery_level"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["next_review_date"] = pd.to_datetime(df["next_review_at"], errors="coerce").dt.date
    df["last_reviewed_date"] = pd.to_datetime(df["last_reviewed_at"], errors="coerce").dt.date
    return df


def merge_terms_into_srs_state(existing_rows: list[dict[str, Any]], terms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now().isoformat(timespec="seconds")
    by_key = {str(r.get("term_key", "")): normalize_srs_row(r) for r in existing_rows if str(r.get("term_key", "")).strip()}
    for term in terms:
        key = term_srs_key(term)
        if not key:
            continue
        script_id = str(term.get("script_id", ""))
        title = str(term.get("title", ""))
        source_file = str(term.get("source_file", ""))
        if key not in by_key:
            by_key[key] = {
                "term_key": key,
                "english": term.get("english", ""),
                "spanish": term.get("spanish", ""),
                "script_ids": script_id,
                "titles": title,
                "source_files": source_file,
                "created_at": now,
                "last_reviewed_at": "",
                "next_review_at": today_date().isoformat(),
                "last_score_label": "",
                "last_score_numeric": "",
                "attempts": "0",
                "misses": "0",
                "lapses": "0",
                "interval_days": "0",
                "mastery_level": "0",
            }
        else:
            row = by_key[key]
            for field, value in [("script_ids", script_id), ("titles", title), ("source_files", source_file)]:
                pieces = [x.strip() for x in str(row.get(field, "")).split(";") if x.strip()]
                if value and value not in pieces:
                    pieces.append(value)
                    row[field] = "; ".join(pieces)
    return list(by_key.values())


def calculate_next_srs_state(row: dict[str, Any], score_label: str) -> dict[str, Any]:
    score = int(TERM_SCORE_OPTIONS.get(score_label, 0))
    attempts = int(float(row.get("attempts", 0) or 0)) + 1
    misses = int(float(row.get("misses", 0) or 0)) + (1 if score <= 0 else 0)
    lapses = int(float(row.get("lapses", 0) or 0)) + (1 if score <= 1 and int(float(row.get("mastery_level", 0) or 0)) >= 2 else 0)
    old_interval = int(float(row.get("interval_days", 0) or 0))
    old_mastery = int(float(row.get("mastery_level", 0) or 0))
    if score <= 0:  # Missed
        interval = 0
        mastery = max(0, old_mastery - 1)
    elif score == 1:  # Hard
        interval = 1 if old_interval <= 1 else max(1, min(3, old_interval // 2))
        mastery = max(0, old_mastery)
    elif score == 2:  # Good
        interval = 3 if old_interval <= 0 else max(3, min(30, int(round(old_interval * 1.8))))
        mastery = min(5, old_mastery + 1)
    else:  # Easy
        interval = 7 if old_interval <= 0 else max(7, min(60, int(round(old_interval * 2.5))))
        mastery = min(5, old_mastery + 2)
    now = datetime.now()
    updated = dict(row)
    updated.update({
        "last_reviewed_at": now.isoformat(timespec="seconds"),
        "next_review_at": (now.date() + timedelta(days=interval)).isoformat(),
        "last_score_label": score_label,
        "last_score_numeric": str(score),
        "attempts": str(attempts),
        "misses": str(misses),
        "lapses": str(lapses),
        "interval_days": str(interval),
        "mastery_level": str(mastery),
    })
    return updated


def rank_terms_for_srs(terms: list[dict[str, Any]], srs_rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if not terms:
        return []
    srs_by_key = {str(r.get("term_key", "")): normalize_srs_row(r) for r in srs_rows}
    today = today_date()
    ranked: list[dict[str, Any]] = []
    for pos, term in enumerate(terms):
        key = term_srs_key(term)
        srs = srs_by_key.get(key, {})
        attempts = int(float(srs.get("attempts", 0) or 0)) if srs else 0
        next_due = parse_date_value(srs.get("next_review_at")) if srs else today
        is_new = attempts == 0
        is_due = is_new or next_due is None or next_due <= today
        overdue_days = 0 if not next_due else max(0, (today - next_due).days)
        score = int(float(srs.get("last_score_numeric", 3) or 3)) if srs else 3
        interval = int(float(srs.get("interval_days", 0) or 0)) if srs else 0
        item = dict(term)
        item.update({
            "term_key": key,
            "srs_attempts": attempts,
            "srs_is_new": is_new,
            "srs_is_due": is_due,
            "srs_next_review_at": srs.get("next_review_at", today.isoformat()) if srs else today.isoformat(),
            "srs_last_score_label": srs.get("last_score_label", "") if srs else "",
            "srs_interval_days": interval,
            "srs_mastery_level": int(float(srs.get("mastery_level", 0) or 0)) if srs else 0,
            "srs_priority": (1000 if is_due else 0) + overdue_days * 10 + (20 if is_new else 0) + max(0, 3 - score) * 8 - min(interval, 30),
            "original_order": pos,
        })
        ranked.append(item)
    if mode == "Due today":
        ranked = [t for t in ranked if t.get("srs_is_due")]
        ranked.sort(key=lambda t: (-int(t.get("srs_priority", 0)), t.get("original_order", 0)))
    elif mode == "New terms":
        ranked = [t for t in ranked if t.get("srs_is_new")]
        ranked.sort(key=lambda t: t.get("original_order", 0))
    elif mode == "Review terms":
        ranked = [t for t in ranked if not t.get("srs_is_new") and t.get("srs_is_due")]
        ranked.sort(key=lambda t: (-int(t.get("srs_priority", 0)), t.get("original_order", 0)))
    elif mode == "All SRS priority":
        ranked.sort(key=lambda t: (-int(t.get("srs_priority", 0)), t.get("original_order", 0)))
    else:
        return rank_terms_for_review(terms, [], mode)
    return ranked



# -----------------------------
# Phase 2D simultaneous practice helpers
# -----------------------------
SIMULTANEOUS_LOG_FIELDS = [
    "timestamp", "session_id", "script_id", "title", "source_file", "practice_mode",
    "training_mode", "lag_seconds", "playback_speed", "repeat_loop",
    "chunk_index", "chunk_label", "source_language", "target_language", "speaker",
    "segment_type", "score", "error_tags", "notes", "source_text", "target_text",
]

SIMULTANEOUS_ERROR_TAGS = [
    "Omission",
    "Lag collapse",
    "False start",
    "Terminology",
    "Register",
    "Numbers/names",
    "Syntax interference",
    "Incomplete thought",
    "Too literal",
    "Lost thread",
]


def simultaneous_log_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=SIMULTANEOUS_LOG_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: row.get(k, "") for k in SIMULTANEOUS_LOG_FIELDS})
    return output.getvalue()


def normalize_simultaneous_log_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {k: row.get(k, "") for k in SIMULTANEOUS_LOG_FIELDS}
    normalized["score"] = str(normalized.get("score", ""))
    return normalized


def load_simultaneous_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [normalize_simultaneous_log_row(row) for row in csv.DictReader(f)]
    except Exception as exc:
        st.warning(f"Could not read simultaneous practice log: {exc}")
        return []


def append_simultaneous_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIMULTANEOUS_LOG_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in SIMULTANEOUS_LOG_FIELDS})


def simultaneous_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SIMULTANEOUS_LOG_FIELDS)
    df = pd.DataFrame([normalize_simultaneous_log_row(r) for r in rows])
    for col in SIMULTANEOUS_LOG_FIELDS:
        if col not in df.columns:
            df[col] = ""
    df["score_numeric"] = pd.to_numeric(df["score"], errors="coerce")
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp_dt"].dt.date.astype("string")
    df["direction"] = df["source_language"].str.title() + " → " + df["target_language"].str.title()
    return df


def build_simultaneous_chunks(data: dict[str, Any], mode: str, direction: str = "Source language first") -> list[dict[str, Any]]:
    paired = data.get("paired_segments", []) or []
    if not isinstance(paired, list):
        paired = []
    script_id = safe_script_id(data, "simultaneous")
    chunks: list[dict[str, Any]] = []
    valid_segments: list[dict[str, Any]] = []
    for idx, seg in enumerate(paired, start=1):
        if not isinstance(seg, dict):
            continue
        source, target = get_segment_source_and_target(seg, direction)
        source_text = str(seg.get(source, "")).strip()
        target_text = str(seg.get(target, "")).strip()
        if not source_text or not target_text:
            continue
        copy = dict(seg)
        copy.update({
            "_segment_index": idx,
            "_source_language": source,
            "_target_language": target,
            "_source_text": source_text,
            "_target_text": target_text,
        })
        valid_segments.append(copy)

    if mode == "Full passage":
        if not valid_segments:
            return []
        source_lang = valid_segments[0]["_source_language"]
        target_lang = valid_segments[0]["_target_language"]
        source_text = "\n\n".join(seg["_source_text"] for seg in valid_segments)
        target_text = "\n\n".join(seg["_target_text"] for seg in valid_segments)
        speakers = sorted({str(seg.get("speaker", "")).strip() for seg in valid_segments if str(seg.get("speaker", "")).strip()})
        segment_types = sorted({str(seg.get("segment_type", "")).strip() for seg in valid_segments if str(seg.get("segment_type", "")).strip()})
        return [{
            "chunk_index": 1,
            "chunk_label": "Full passage",
            "source_language": source_lang,
            "target_language": target_lang,
            "source_text": source_text,
            "target_text": target_text,
            "speaker": ", ".join(speakers[:6]),
            "segment_type": ", ".join(segment_types[:6]),
            "segments": valid_segments,
        }]

    for pos, seg in enumerate(valid_segments, start=1):
        chunks.append({
            "chunk_index": pos,
            "chunk_label": f"Chunk {pos} / JSON segment {seg.get('_segment_index')}",
            "source_language": seg["_source_language"],
            "target_language": seg["_target_language"],
            "source_text": seg["_source_text"],
            "target_text": seg["_target_text"],
            "speaker": str(seg.get("speaker", "")),
            "segment_type": str(seg.get("segment_type", "")),
            "segments": [seg],
        })
    return chunks


def get_simultaneous_audio_path(cache_dir: Path, script_id: str, chunk_index: Any, role: str, lang: str, text: str, voice: str, rate: str) -> Path:
    key = short_hash(script_id, str(chunk_index), role, lang, text, voice, rate)
    filename = f"{sanitize_filename_part(script_id)}_sim_{sanitize_filename_part(str(chunk_index), 20)}_{role}_{lang}_{key}.mp3"
    return cache_dir / filename


def reset_simultaneous_state(reason: str = "") -> None:
    st.session_state.simultaneous_idx = 0
    st.session_state.simultaneous_revealed = False
    st.session_state.simultaneous_signature = None
    st.session_state.simultaneous_source_nonce = 0
    st.session_state.simultaneous_target_nonce = 0
    st.session_state.simultaneous_started_at = time.time()
    st.session_state.simultaneous_lag_started = False
    st.session_state.simultaneous_lag_start_time = None
    st.session_state.simultaneous_reason = reason
    st.session_state.simultaneous_session_id = f"sim-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

def reset_term_review_state(reason: str = "") -> None:
    st.session_state.term_review_idx = 0
    st.session_state.term_review_revealed = False
    st.session_state.term_review_started_at = time.time()
    st.session_state.term_review_reason = reason
    st.session_state.term_review_session_id = f"term-session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    st.session_state.term_review_shuffle_seed = uuid.uuid4().hex[:8]

def reset_practice_state(reason: str = "") -> None:
    st.session_state.practice_idx = 0
    st.session_state.practice_revealed = False
    st.session_state.practice_segment_signature = None
    st.session_state.source_replay_nonce = 0
    st.session_state.target_replay_nonce = 0
    st.session_state.target_reveal_nonce = 0
    st.session_state.practice_started_at = time.time()
    st.session_state.practice_reason = reason
    st.session_state.practice_session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"

st.set_page_config(page_title="Interpreter Audio Generator", page_icon="🎧", layout="wide")
inject_phase2e_css()
st.title("🎧 Interpreter Audio Generator — Web 1.2")
st.caption("Cloud-safe Streamlit app with audio generation, consecutive practice, simultaneous practice, term review, spoken terms, spaced review, and cleaner side-by-side practice views.")
st.info("Cloud version note: progress and cached audio are temporary for this app session. Download your logs/files if you want to keep them.")

with st.sidebar:
    st.header("Generator")
    generator_path = DEFAULT_GENERATOR_PATH
    output_dir = RUNTIME_OUTPUT_DIR
    st.caption("Cloud mode: files are generated temporarily and offered as downloads.")
    st.divider()
    st.header("Audio settings")
    speed_mode = st.selectbox("Speed mode", ["learning", "normal", "fast"], index=0)
    split_flashcards = st.checkbox("Split flashcards and script", value=True)
    include_flashcards = st.checkbox("Include flashcards", value=True)
    consecutive_mode = st.checkbox("Generate consecutive mode", value=True)
    full_speed = st.checkbox("Generate full speed-increase drill", value=True)
    targeted_terms = st.checkbox("Generate targeted hard-terms drill", value=False)
    inline_cues = st.checkbox("Generate inline term-cue drill", value=False)
    combined_all = st.checkbox("Generate combined audio from all script outputs", value=False)
    skip_existing = st.checkbox("Skip existing up-to-date MP3s", value=True)
    verify_voices = st.checkbox("Verify Edge TTS voices", value=True)

    st.subheader("Consecutive mode timing")
    consecutive_ratio = st.number_input("Pause ratio", min_value=0.1, max_value=5.0, value=1.35, step=0.05)
    consecutive_min = st.number_input("Minimum pause", min_value=0.0, max_value=60.0, value=3.0, step=0.5)
    consecutive_max = st.number_input("Maximum pause", min_value=0.5, max_value=120.0, value=14.0, step=0.5)
    consecutive_gap = st.number_input("Segment gap", min_value=0.0, max_value=20.0, value=1.8, step=0.1)

    st.subheader("Interactive practice audio")
    practice_audio_enabled = st.checkbox("Enable practice audio", value=True)
    practice_audio_autoplay_source = st.checkbox("Auto-play source audio when segment loads", value=True)
    practice_audio_play_correction = st.checkbox("Play correction audio when revealing target", value=True)
    practice_audio_show_replay = st.checkbox("Show replay audio controls", value=True)
    practice_audio_use_profile = st.checkbox("Use JSON audio_profile speaker voices", value=True)
    audio_only_source_mode = st.checkbox("Consecutive audio-only source mode", value=False)
    consecutive_playback_speed = st.selectbox(
        "Consecutive playback speed",
        [0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50],
        index=1,
        format_func=lambda x: f"{x:.2f}x",
        key="consecutive_playback_speed",
        help="Applies to both the source audio and the revealed correction audio in Consecutive Practice.",
    )
    practice_audio_cache_dir = RUNTIME_CACHE_DIR / "practice_audio_cache"


    st.divider()
    st.header("Study progress")
    persistent_log_path = DEFAULT_PRACTICE_LOG_PATH
    auto_save_practice_log = st.checkbox("Auto-save scored segments to study history", value=True)
    simultaneous_log_path = DEFAULT_SIMULTANEOUS_LOG_PATH
    auto_save_simultaneous_log = st.checkbox("Auto-save simultaneous practice attempts", value=True)
    term_review_log_path = DEFAULT_TERM_REVIEW_LOG_PATH
    term_srs_state_path = DEFAULT_TERM_SRS_STATE_PATH
    auto_save_term_review_log = st.checkbox("Auto-save term review scores", value=True)

    st.subheader("Spoken term review audio")
    term_audio_enabled = st.checkbox("Enable term review audio", value=True)
    term_audio_autoplay_prompt = st.checkbox("Auto-play prompt term", value=True)
    term_audio_play_answer_on_reveal = st.checkbox("Play answer audio when revealing target", value=True)
    term_audio_show_replay = st.checkbox("Show term replay audio controls", value=True)
    term_audio_only_prompt = st.checkbox("Audio-only prompt mode", value=False)
    term_audio_cache_dir = RUNTIME_CACHE_DIR / "term_audio_cache"

    show_history_preview_rows = st.number_input("History rows to preview", min_value=10, max_value=1000, value=100, step=10)
    st.caption("Cloud logs are temporary. Use the download buttons to keep a copy.")

tab_generate, tab_practice, tab_simul, tab_history, tab_terms = st.tabs(["Generate outputs", "Consecutive practice", "Simultaneous practice", "Study history", "Term review"])

with tab_generate:
    st.subheader("1. Choose source JSON files")
    loaded_files: list[tuple[str, bytes]] = []
    uploads = st.file_uploader("Upload one or more JSON scripts", type=["json"], accept_multiple_files=True)
    for uploaded in uploads or []:
        loaded_files.append((uploaded.name, uploaded.getvalue()))

    payloads: list[tuple[str, dict[str, Any], bytes]] = []
    parse_errors: list[str] = []
    for name, raw in loaded_files:
        data, error = load_json_bytes(name, raw)
        if error:
            parse_errors.append(error)
        elif data is not None:
            payloads.append((name, data, raw))

    if parse_errors:
        for err in parse_errors:
            st.error(err)

    if payloads:
        st.subheader("2. Preview and validate")
        summaries = [summarize_payload(data, name) for name, data, _ in payloads]
        st.dataframe(summaries, use_container_width=True, hide_index=True)

        with st.expander("Validation details", expanded=True):
            for name, data, _ in payloads:
                warnings, errors = validate_json_payload(data)
                label = f"{name}: {len(errors)} error(s), {len(warnings)} warning(s)"
                with st.expander(label, expanded=bool(errors)):
                    if errors:
                        for err in errors:
                            st.error(err)
                    if warnings:
                        for warning in warnings[:50]:
                            st.warning(warning)
                        if len(warnings) > 50:
                            st.info(f"Showing first 50 warnings out of {len(warnings)}.")
                    if not errors and not warnings:
                        st.success("No validation issues found.")

        st.subheader("3. Run")
        settings = {
            "SPEED_MODE": speed_mode,
            "SPLIT_FLASHCARDS_AND_SCRIPT": split_flashcards,
            "INCLUDE_FLASHCARDS": include_flashcards,
            "GENERATE_CONSECUTIVE_MODE": consecutive_mode,
            "CONSECUTIVE_PAUSE_RATIO": float(consecutive_ratio),
            "CONSECUTIVE_MIN_PAUSE": float(consecutive_min),
            "CONSECUTIVE_MAX_PAUSE": float(consecutive_max),
            "CONSECUTIVE_SEGMENT_GAP": float(consecutive_gap),
            "GENERATE_FULL_SPEED_INCREASE": full_speed,
            "GENERATE_TARGETED_HARD_TERMS": targeted_terms,
            "GENERATE_INLINE_TERM_CUES": inline_cues,
            "GENERATE_COMBINED_AUDIO_ALL_SCRIPTS": combined_all,
            "SKIP_EXISTING_SCRIPT_MP3S": skip_existing,
            "VERIFY_TTS_VOICES": verify_voices,
        }

        disabled = not generator_path.exists() or not payloads
        if not generator_path.exists():
            st.error("Generator script path does not exist.")

        if st.button("Run audio generator", type="primary", disabled=disabled):
            output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="streamlit_audio_json_") as tmp:
                tmpdir = Path(tmp)
                json_paths: list[Path] = []
                for name, _data, raw in payloads:
                    path = tmpdir / Path(name).name
                    path.write_bytes(raw)
                    json_paths.append(path)

                log_buffer = io.StringIO()
                try:
                    with st.spinner("Generating audio..."):
                        with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                            outputs = asyncio.run(run_generator(generator_path, json_paths, output_dir, settings))
                    st.success(f"Generated {len(outputs)} output file(s).")
                    if outputs:
                        st.write("Output files generated for this session:")
                        for p in outputs:
                            st.code(p.name)
                        zip_bytes = make_zip_bytes([Path(p) for p in outputs], output_dir)
                        st.download_button(
                            "Download generated outputs ZIP",
                            data=zip_bytes,
                            file_name=f"interpreter_generated_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                        )
                    with st.expander("Generator log", expanded=False):
                        st.text(log_buffer.getvalue() or "No log output.")
                except Exception as exc:
                    st.error(f"Generator failed: {exc}")
                    with st.expander("Generator log", expanded=True):
                        st.text(log_buffer.getvalue() or "No log output.")
    else:
        st.info("Choose one or more JSON files to preview settings and run the generator.")

with tab_practice:
    st.subheader("Interactive consecutive practice")
    st.caption("Practice segment by segment: read/listen to the source side, interpret aloud, reveal the correction, score yourself, and automatically save long-term progress.")
    st.caption(f"Current practice session: `{st.session_state.get('practice_session_id', 'not started')}`")

    with st.expander("Consecutive audio controls", expanded=True):
        control_cols = st.columns([1, 1])
        with control_cols[0]:
            consecutive_playback_speed = st.selectbox(
                "Playback speed",
                [0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50],
                index=[0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50].index(float(st.session_state.get("consecutive_playback_speed", 1.0))) if float(st.session_state.get("consecutive_playback_speed", 1.0)) in [0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50] else 1,
                format_func=lambda x: f"{x:.2f}x",
                key="consecutive_playback_speed_inline",
                help="Applies to both source audio and correction/target audio.",
            )
        with control_cols[1]:
            audio_only_source_mode = st.checkbox(
                "Audio-only source mode",
                value=bool(st.session_state.get("consecutive_audio_only_inline", False)),
                key="consecutive_audio_only_inline",
                help="Hide the source transcript until you want to practice from audio only.",
            )

    if "practice_log" not in st.session_state:
        st.session_state.practice_log = []
    if "practice_session_id" not in st.session_state:
        st.session_state.practice_session_id = f"session-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    if "practice_idx" not in st.session_state:
        reset_practice_state("initial")

    practice_files: list[tuple[str, bytes]] = []
    practice_source = st.radio("Practice source", ["Upload JSON file", "Use selected files from Generate tab"], horizontal=True)
    if practice_source == "Upload JSON file":
        uploaded_practice = st.file_uploader("Choose one JSON for practice", type=["json"], accept_multiple_files=False, key="practice_upload")
        if uploaded_practice:
            practice_files.append((uploaded_practice.name, uploaded_practice.getvalue()))
    elif practice_source == "Use selected files from Generate tab":
        practice_files = loaded_files[:]
        if not practice_files:
            st.info("Select files in the Generate outputs tab first, or use Upload JSON file here.")
    else:
        practice_path = Path(st.text_input("Practice JSON path", value="")).expanduser()
        if practice_path.exists() and practice_path.suffix.lower() == ".json":
            practice_files.append((practice_path.name, practice_path.read_bytes()))
        elif str(practice_path):
            st.warning("That JSON path does not exist.")

    practice_payloads: list[tuple[str, dict[str, Any]]] = []
    for name, raw in practice_files:
        data, error = load_json_bytes(name, raw)
        if error:
            st.error(error)
        elif data:
            practice_payloads.append((name, data))

    if practice_payloads:
        file_options = [f"{safe_script_id(data, Path(name).stem)} — {data.get('title', name)}" for name, data in practice_payloads]
        selected_label = st.selectbox("Script", file_options)
        selected_idx = file_options.index(selected_label)
        selected_name, selected_data = practice_payloads[selected_idx]
        script_id = safe_script_id(selected_data, Path(selected_name).stem)
        title = str(selected_data.get("title", selected_name))

        paired = selected_data.get("paired_segments", []) or []
        if not isinstance(paired, list) or not paired:
            st.warning("This JSON does not contain paired_segments for interactive practice.")
        else:
            all_types = sorted({str(seg.get("segment_type", "")).strip() for seg in paired if isinstance(seg, dict) and str(seg.get("segment_type", "")).strip()})
            all_speakers = sorted({str(seg.get("speaker", "")).strip() for seg in paired if isinstance(seg, dict) and str(seg.get("speaker", "")).strip()})

            c1, c2, c3 = st.columns(3)
            with c1:
                direction = st.selectbox("Practice direction", ["Source language first", "English → Spanish only", "Spanish → English only"])
            with c2:
                selected_types = st.multiselect("Segment types", all_types, default=[])
            with c3:
                selected_speakers = st.multiselect("Speakers", all_speakers, default=[])

            randomize = st.checkbox("Randomize segment order", value=False)
            segments = filter_practice_segments(selected_data, direction, selected_types, selected_speakers)
            if randomize:
                # Stable deterministic randomization per script/load so Streamlit reruns do not jump unexpectedly.
                import random
                rng = random.Random(script_id + title + direction)
                segments = segments[:]
                rng.shuffle(segments)

            if not segments:
                st.warning("No segments match the current filters.")
            else:
                st.write(f"Practice set: **{len(segments)} segment(s)**")
                if st.button("Start / reset this practice set"):
                    reset_practice_state("manual reset")

                st.session_state.practice_idx = min(st.session_state.practice_idx, len(segments) - 1)
                idx = st.session_state.practice_idx
                seg = segments[idx]
                source_lang = seg["_source_language"]
                target_lang = seg["_target_language"]
                source_text = str(seg.get(source_lang, "")).strip()
                target_text = str(seg.get(target_lang, "")).strip()

                # Detect an actual segment change and reset audio/reveal state. This prevents
                # the source player from reusing the previous segment's browser audio element.
                segment_signature = short_hash(
                    script_id,
                    str(idx),
                    str(seg.get("_segment_index")),
                    source_lang,
                    target_lang,
                    source_text,
                    target_text,
                )
                if st.session_state.get("practice_segment_signature") != segment_signature:
                    st.session_state.practice_segment_signature = segment_signature
                    st.session_state.practice_revealed = False
                    st.session_state.source_replay_nonce = st.session_state.get("source_replay_nonce", 0) + 1
                    st.session_state.target_replay_nonce = 0
                    st.session_state.target_reveal_nonce = 0

                pause_estimate = estimate_dynamic_pause_seconds(source_text, float(consecutive_ratio), float(consecutive_min), float(consecutive_max))

                st.progress((idx + 1) / len(segments))
                st.markdown(f"### Segment {idx + 1} of {len(segments)}")
                render_compact_segment_meta(
                    primary_label=f"JSON segment {seg.get('_segment_index')}",
                    source_lang=source_lang,
                    target_lang=target_lang,
                    word_count=len(source_text.split()),
                    speaker=str(seg.get("speaker") or ""),
                    segment_type=str(seg.get("segment_type") or ""),
                    extra=f"pause ~{pause_estimate}s",
                )

                st.markdown("#### Practice workspace")
                render_side_by_side_text(
                    source_text=source_text,
                    target_text=target_text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    speaker=str(seg.get('speaker', '')),
                    segment_type=str(seg.get('segment_type', '')),
                    revealed=bool(st.session_state.practice_revealed),
                    source_visible=not bool(audio_only_source_mode),
                    context_label="Correction / target",
                )
                st.caption("Interpret this aloud before revealing the correction.")

                source_audio_path = None
                target_audio_path = None
                if practice_audio_enabled:
                    source_voice = get_practice_voice(selected_data, seg, source_lang, practice_audio_use_profile)
                    target_voice = get_practice_voice(selected_data, seg, target_lang, practice_audio_use_profile)
                    source_rate = get_rate_for_language(speed_mode, source_lang)
                    target_rate = get_rate_for_language(speed_mode, target_lang)
                    source_audio_path = get_practice_audio_path(practice_audio_cache_dir, script_id, seg.get("_segment_index", idx + 1), "source", source_lang, source_text, source_voice, source_rate)
                    target_audio_path = get_practice_audio_path(practice_audio_cache_dir, script_id, seg.get("_segment_index", idx + 1), "target", target_lang, target_text, target_voice, target_rate)

                    try:
                        with st.spinner("Preparing source audio..."):
                            asyncio.run(generate_practice_audio_file(source_text, source_voice, source_rate, source_audio_path, generator_path))
                        audio_cols = st.columns([2, 1, 1])
                        with audio_cols[0]:
                            render_audio_player(
                                source_audio_path,
                                label=f"Source audio — segment {seg.get('_segment_index')} — {source_lang.title()} — voice: {source_voice}",
                                autoplay=practice_audio_autoplay_source and not st.session_state.practice_revealed,
                                show_native_player=practice_audio_show_replay,
                                playback_speed=float(consecutive_playback_speed),
                                nonce=f"source-{segment_signature}-{st.session_state.get('source_replay_nonce', 0)}",
                            )
                        with audio_cols[1]:
                            if practice_audio_show_replay and st.button("Replay source audio", key=f"replay_source_{segment_signature}"):
                                st.session_state.source_replay_nonce = st.session_state.get("source_replay_nonce", 0) + 1
                                st.rerun()
                        with audio_cols[2]:
                            st.caption(f"Source voice: {source_voice}")
                    except Exception as exc:
                        st.warning(f"Could not prepare source audio: {exc}")

                reveal_clicked = st.button("Reveal correction / target text", type="primary")
                if reveal_clicked:
                    st.session_state.practice_revealed = True
                    st.session_state.target_reveal_nonce = st.session_state.get("target_reveal_nonce", 0) + 1
                    st.rerun()

                if st.session_state.practice_revealed:
                    st.caption("Correction is shown in the right-hand reference panel above.")

                    if practice_audio_enabled and target_audio_path is not None:
                        try:
                            target_voice = get_practice_voice(selected_data, seg, target_lang, practice_audio_use_profile)
                            target_rate = get_rate_for_language(speed_mode, target_lang)
                            with st.spinner("Preparing correction audio..."):
                                asyncio.run(generate_practice_audio_file(target_text, target_voice, target_rate, target_audio_path, generator_path))
                            correction_cols = st.columns([2, 1, 1])
                            with correction_cols[0]:
                                render_audio_player(
                                    target_audio_path,
                                    label=f"Correction audio — segment {seg.get('_segment_index')} — {target_lang.title()} — voice: {target_voice}",
                                    autoplay=practice_audio_play_correction,
                                    show_native_player=practice_audio_show_replay,
                                    playback_speed=float(consecutive_playback_speed),
                                    nonce=f"target-{segment_signature}-{st.session_state.get('target_reveal_nonce', 0)}-{st.session_state.get('target_replay_nonce', 0)}",
                                )
                            with correction_cols[1]:
                                if practice_audio_show_replay and st.button("Replay correction audio", key=f"replay_target_{segment_signature}"):
                                    st.session_state.target_replay_nonce = st.session_state.get("target_replay_nonce", 0) + 1
                                    st.rerun()
                            with correction_cols[2]:
                                st.caption(f"Correction voice: {target_voice}")
                        except Exception as exc:
                            st.warning(f"Could not prepare correction audio: {exc}")

                    st.markdown("#### Self-score")
                    score = st.radio(
                        "Score",
                        options=[3, 2, 1, 0],
                        format_func=lambda x: {
                            3: "3 — Accurate",
                            2: "2 — Minor issue",
                            1: "1 — Major issue",
                            0: "0 — Missed / could not complete",
                        }[x],
                        horizontal=True,
                        key=f"score_{script_id}_{idx}_{seg.get('_segment_index')}",
                    )
                    error_tags = st.multiselect("Error tags", ERROR_TAGS, key=f"tags_{script_id}_{idx}_{seg.get('_segment_index')}")
                    notes = st.text_area("Notes", key=f"notes_{script_id}_{idx}_{seg.get('_segment_index')}", height=90)

                    col_prev, col_save, col_next = st.columns([1, 2, 1])
                    with col_prev:
                        if st.button("← Previous", disabled=idx == 0):
                            st.session_state.practice_idx -= 1
                            st.session_state.practice_revealed = False
                            st.rerun()
                    with col_save:
                        if st.button("Save score and next", type="primary"):
                            log_row = {
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "session_id": st.session_state.get("practice_session_id", ""),
                                "script_id": script_id,
                                "title": title,
                                "practice_direction": direction,
                                "practice_set_size": len(segments),
                                "segment_index": seg.get("_segment_index"),
                                "speaker": seg.get("speaker", ""),
                                "segment_type": seg.get("segment_type", ""),
                                "source_language": source_lang,
                                "target_language": target_lang,
                                "score": score,
                                "error_tags": "; ".join(error_tags),
                                "notes": notes,
                                "source_text": source_text,
                                "target_text": target_text,
                            }
                            st.session_state.practice_log.append(log_row)
                            if auto_save_practice_log:
                                try:
                                    append_persistent_practice_log(persistent_log_path, log_row)
                                except Exception as exc:
                                    st.warning(f"Could not auto-save to study history: {exc}")
                            if idx < len(segments) - 1:
                                st.session_state.practice_idx += 1
                                st.session_state.practice_revealed = False
                            st.rerun()
                    with col_next:
                        if st.button("Skip →", disabled=idx >= len(segments) - 1):
                            st.session_state.practice_idx += 1
                            st.session_state.practice_revealed = False
                            st.rerun()
                else:
                    nav1, nav2 = st.columns([1, 1])
                    with nav1:
                        if st.button("← Previous segment", disabled=idx == 0):
                            st.session_state.practice_idx -= 1
                            st.session_state.practice_revealed = False
                            st.rerun()
                    with nav2:
                        if st.button("Skip segment →", disabled=idx >= len(segments) - 1):
                            st.session_state.practice_idx += 1
                            st.session_state.practice_revealed = False
                            st.rerun()

                st.divider()
                st.markdown("### Session log")
                log_rows = st.session_state.practice_log
                if log_rows:
                    st.dataframe(log_rows, use_container_width=True, hide_index=True)
                    scores = [int(r.get("score", 0)) for r in log_rows if str(r.get("score", "")).isdigit()]
                    if scores:
                        st.write(f"Average score: **{sum(scores) / len(scores):.2f} / 3** across **{len(scores)} saved segment(s)**.")
                    if auto_save_practice_log:
                        st.caption(f"Scored segments are being saved to: `{persistent_log_path}`")
                    csv_text = practice_log_to_csv(log_rows)
                    st.download_button(
                        "Download session log CSV",
                        data=csv_text.encode("utf-8"),
                        file_name=f"{script_id}_practice_log.csv",
                        mime="text/csv",
                    )
                    if st.button("Clear session log"):
                        st.session_state.practice_log = []
                        st.rerun()
                else:
                    st.info("No scores saved yet in this app session.")
    else:
        st.info("Choose a JSON file to start interactive consecutive practice.")


with tab_simul:
    st.subheader("Simultaneous practice")
    st.caption("Listen to a continuous source passage, interpret aloud in real time, reveal the reference interpretation, then score and save your attempt.")
    st.caption(f"Current simultaneous session: `{st.session_state.get('simultaneous_session_id', 'not started')}`")

    if "simultaneous_log" not in st.session_state:
        st.session_state.simultaneous_log = []
    if "simultaneous_session_id" not in st.session_state:
        reset_simultaneous_state("initial")

    sim_files: list[tuple[str, bytes]] = []
    sim_source = st.radio("Simultaneous source", ["Upload JSON file", "Use selected files from Generate tab"], horizontal=True, key="sim_source_radio")
    if sim_source == "Upload JSON file":
        uploaded_sim = st.file_uploader("Choose one simultaneous JSON", type=["json"], accept_multiple_files=False, key="sim_upload")
        if uploaded_sim:
            sim_files.append((uploaded_sim.name, uploaded_sim.getvalue()))
    elif sim_source == "Use selected files from Generate tab":
        sim_files = loaded_files[:]
        if not sim_files:
            st.info("Select files in the Generate outputs tab first, or upload a JSON here.")
    else:
        sim_path = Path(st.text_input("Simultaneous JSON path", value="", key="sim_path")).expanduser()
        if sim_path.exists() and sim_path.suffix.lower() == ".json":
            sim_files.append((sim_path.name, sim_path.read_bytes()))
        elif str(sim_path):
            st.warning("That JSON path does not exist.")

    sim_payloads: list[tuple[str, dict[str, Any]]] = []
    for name, raw in sim_files:
        data, error = load_json_bytes(name, raw)
        if error:
            st.error(error)
        elif data:
            sim_payloads.append((name, data))

    if sim_payloads:
        sim_options = [f"{safe_script_id(data, Path(name).stem)} — {data.get('title', name)}" for name, data in sim_payloads]
        sim_label = st.selectbox("Script", sim_options, key="sim_script_select")
        sim_selected_idx = sim_options.index(sim_label)
        sim_name, sim_data = sim_payloads[sim_selected_idx]
        sim_script_id = safe_script_id(sim_data, Path(sim_name).stem)
        sim_title = str(sim_data.get("title", sim_name))

        format_hint = str(sim_data.get("format", "")).lower()
        if "simultaneous" not in format_hint:
            st.info("This file does not say 'simultaneous' in its format field, but it can still be practiced here if it has paired_segments.")

        settings_cols = st.columns(4)
        with settings_cols[0]:
            sim_mode = st.selectbox("Practice chunk size", ["Paragraph / thought chunk", "Full passage"], key="sim_mode")
        with settings_cols[1]:
            sim_direction = st.selectbox("Direction", ["Source language first", "English → Spanish only", "Spanish → English only"], key="sim_direction")
        with settings_cols[2]:
            sim_audio_only = st.checkbox("Audio-only source mode", value=True, key="sim_audio_only")
        with settings_cols[3]:
            show_source_transcript = st.checkbox("Show source transcript", value=not sim_audio_only, key="sim_show_source")

        reveal_reference_audio = st.checkbox("Play reference audio when revealing target", value=True, key="sim_play_ref")
        show_reference_text_by_default = st.checkbox("Show reference text after reveal", value=True, key="sim_show_ref_text")

        with st.expander("Phase 2D.1 simultaneous training tools", expanded=True):
            tool_cols = st.columns(4)
            with tool_cols[0]:
                sim_training_mode = st.selectbox(
                    "Training mode",
                    ["Interpret", "Lag trainer", "Shadowing", "Shadow first, then interpret"],
                    key="sim_training_mode",
                )
            with tool_cols[1]:
                sim_lag_seconds = st.number_input("Lag target seconds", min_value=0, max_value=15, value=3, step=1, key="sim_lag_seconds")
            with tool_cols[2]:
                sim_playback_speed = st.selectbox("Playback speed", [0.85, 1.0, 1.10, 1.20, 1.30, 1.40, 1.50], index=1, format_func=lambda x: f"{x:.2f}x", key="sim_playback_speed")
            with tool_cols[3]:
                sim_repeat_loop = st.checkbox("Repeat-loop source audio", value=False, key="sim_repeat_loop")

            st.caption(
                "Lag trainer gives you a start cue after the selected delay. "
                "Shadowing means repeating the source language first; Shadow first, then interpret adds an interpretation pass after shadowing."
            )

        chunk_mode_internal = "Full passage" if sim_mode == "Full passage" else "Paragraph / thought chunk"
        chunks = build_simultaneous_chunks(sim_data, chunk_mode_internal, sim_direction)
        if not chunks:
            st.warning("No usable paired_segments were found for simultaneous practice.")
        else:
            st.write(f"Practice set: **{len(chunks)} chunk(s)**")
            if st.button("Start / reset simultaneous practice", key="sim_reset"):
                reset_simultaneous_state("manual reset")
                st.rerun()

            st.session_state.simultaneous_idx = min(st.session_state.simultaneous_idx, len(chunks) - 1)
            sim_idx = st.session_state.simultaneous_idx
            chunk = chunks[sim_idx]
            source_lang = str(chunk.get("source_language", "english"))
            target_lang = str(chunk.get("target_language", "spanish"))
            source_text = str(chunk.get("source_text", "")).strip()
            target_text = str(chunk.get("target_text", "")).strip()
            chunk_signature = short_hash(sim_script_id, str(sim_idx), chunk.get("chunk_label", ""), source_lang, target_lang, source_text, target_text)
            if st.session_state.get("simultaneous_signature") != chunk_signature:
                st.session_state.simultaneous_signature = chunk_signature
                st.session_state.simultaneous_revealed = False
                st.session_state.simultaneous_source_nonce = st.session_state.get("simultaneous_source_nonce", 0) + 1
                st.session_state.simultaneous_target_nonce = 0
                st.session_state.simultaneous_lag_started = False
                st.session_state.simultaneous_lag_start_time = None

            st.progress((sim_idx + 1) / len(chunks), text=f"Chunk {sim_idx + 1} of {len(chunks)}")
            chunk_label = chunk.get('chunk_label', f'Chunk {sim_idx + 1}')
            st.markdown(f"### {chunk_label}")
            render_compact_segment_meta(
                primary_label=str(chunk_label),
                source_lang=source_lang,
                target_lang=target_lang,
                word_count=len(source_text.split()),
                speaker=str(chunk.get("speaker") or ""),
                segment_type=str(chunk.get("segment_type") or ""),
            )

            # Use audio_profile speaker assignments for both paragraph chunks and full-passage mode.
            # In full-passage mode, a single MP3 can only use one voice, so the helper uses
            # the first available segment's assigned speaker voice instead of falling back
            # to the global default. This preserves the assigned voice for monologues.
            source_voice = get_practice_voice_for_chunk(sim_data, chunk, source_lang, practice_audio_use_profile)
            target_voice = get_practice_voice_for_chunk(sim_data, chunk, target_lang, practice_audio_use_profile)
            source_rate = get_rate_for_language(speed_mode, source_lang)
            target_rate = get_rate_for_language(speed_mode, target_lang)
            sim_cache_dir = practice_audio_cache_dir / "simultaneous"
            source_audio_path = get_simultaneous_audio_path(sim_cache_dir, sim_script_id, chunk.get("chunk_index", sim_idx + 1), "source", source_lang, source_text, source_voice, source_rate)
            target_audio_path = get_simultaneous_audio_path(sim_cache_dir, sim_script_id, chunk.get("chunk_index", sim_idx + 1), "reference", target_lang, target_text, target_voice, target_rate)

            st.markdown("#### Source audio")
            if practice_audio_enabled:
                try:
                    with st.spinner("Preparing simultaneous source audio..."):
                        asyncio.run(generate_practice_audio_file(source_text, source_voice, source_rate, source_audio_path, generator_path))
                    render_simultaneous_audio_player(
                        source_audio_path,
                        label=f"Simultaneous source — {source_lang.title()} — voice: {source_voice}",
                        autoplay=practice_audio_autoplay_source and not st.session_state.simultaneous_revealed,
                        show_native_player=True,
                        nonce=f"sim-source-{chunk_signature}-{st.session_state.get('simultaneous_source_nonce', 0)}",
                        playback_speed=float(sim_playback_speed),
                        loop=bool(sim_repeat_loop),
                    )
                    if st.button("↻ Replay source", key=f"sim_replay_source_{chunk_signature}"):
                        st.session_state.simultaneous_source_nonce = st.session_state.get("simultaneous_source_nonce", 0) + 1
                        st.rerun()
                except Exception as exc:
                    st.warning(f"Could not prepare source audio: {exc}")
            else:
                st.info("Practice audio is disabled in the sidebar. Enable practice audio to use this tab as intended.")

            st.markdown("#### Phase 2D.1 practice cue")
            if sim_training_mode == "Lag trainer":
                cue_cols = st.columns([1, 1, 2])
                with cue_cols[0]:
                    if st.button("Start lag timer", key=f"sim_start_lag_{chunk_signature}"):
                        st.session_state.simultaneous_lag_started = True
                        st.session_state.simultaneous_lag_start_time = time.time()
                        st.rerun()
                with cue_cols[1]:
                    if st.button("Reset lag timer", key=f"sim_reset_lag_{chunk_signature}"):
                        st.session_state.simultaneous_lag_started = False
                        st.session_state.simultaneous_lag_start_time = None
                        st.rerun()
                lag_start = st.session_state.get("simultaneous_lag_start_time")
                if st.session_state.get("simultaneous_lag_started") and lag_start:
                    elapsed = time.time() - float(lag_start)
                    remaining = max(0, int(sim_lag_seconds) - elapsed)
                    if remaining > 0:
                        st.warning(f"Hold your output. Begin interpreting in approximately {remaining:.1f} second(s).")
                    else:
                        st.success("Begin interpreting now. Maintain the lag while the source continues.")
                else:
                    st.info("Start the source audio, then press Start lag timer. Begin interpreting when the cue appears.")
            elif sim_training_mode == "Shadowing":
                st.info("Shadowing mode: repeat the source language aloud while listening. Reveal the reference only after the shadowing pass.")
            elif sim_training_mode == "Shadow first, then interpret":
                st.info("First pass: shadow the source language. Replay the source, then interpret into the target language on the second pass.")
            else:
                st.info("Interpret aloud in real time while the source audio plays.")

            st.markdown("#### Practice workspace")
            render_side_by_side_text(
                source_text=source_text,
                target_text=target_text,
                source_lang=source_lang,
                target_lang=target_lang,
                speaker=str(chunk.get('speaker') or ''),
                segment_type=str(chunk.get('segment_type') or ''),
                revealed=bool(st.session_state.simultaneous_revealed) and bool(show_reference_text_by_default),
                source_visible=bool(show_source_transcript),
                context_label="Reference interpretation",
            )

            st.markdown("#### Practice task")
            st.write("Use the selected training mode above. When finished, reveal the reference interpretation and score your performance.")

            if not st.session_state.simultaneous_revealed:
                if st.button("Reveal reference interpretation", type="primary", key=f"sim_reveal_{chunk_signature}"):
                    st.session_state.simultaneous_revealed = True
                    st.session_state.simultaneous_target_nonce = st.session_state.get("simultaneous_target_nonce", 0) + 1
                    st.rerun()
            else:
                if show_reference_text_by_default:
                    st.caption("Reference interpretation is shown in the right-hand panel above.")
                else:
                    with st.expander("Reference interpretation text", expanded=False):
                        st.success(target_text)

                if practice_audio_enabled:
                    try:
                        with st.spinner("Preparing reference audio..."):
                            asyncio.run(generate_practice_audio_file(target_text, target_voice, target_rate, target_audio_path, generator_path))
                        render_simultaneous_audio_player(
                            target_audio_path,
                            label=f"Reference interpretation — {target_lang.title()} — voice: {target_voice}",
                            autoplay=reveal_reference_audio,
                            show_native_player=True,
                            nonce=f"sim-target-{chunk_signature}-{st.session_state.get('simultaneous_target_nonce', 0)}",
                            playback_speed=float(sim_playback_speed),
                            loop=False,
                        )
                        if st.button("↻ Replay reference", key=f"sim_replay_target_{chunk_signature}"):
                            st.session_state.simultaneous_target_nonce = st.session_state.get("simultaneous_target_nonce", 0) + 1
                            st.rerun()
                    except Exception as exc:
                        st.warning(f"Could not prepare reference audio: {exc}")

                st.markdown("#### Self-score")
                sim_score = st.radio(
                    "Score",
                    options=[3, 2, 1, 0],
                    format_func=lambda x: {3: "3 — Strong", 2: "2 — Usable with minor issues", 1: "1 — Major issues", 0: "0 — Lost / could not complete"}[x],
                    horizontal=True,
                    key=f"sim_score_{chunk_signature}",
                )
                sim_tags = st.multiselect("Simultaneous issue tags", SIMULTANEOUS_ERROR_TAGS, key=f"sim_tags_{chunk_signature}")
                sim_notes = st.text_area("Notes", key=f"sim_notes_{chunk_signature}", height=90)

                nav_cols = st.columns([1, 2, 1])
                with nav_cols[0]:
                    if st.button("← Previous chunk", disabled=sim_idx == 0, key=f"sim_prev_{chunk_signature}"):
                        st.session_state.simultaneous_idx -= 1
                        st.session_state.simultaneous_revealed = False
                        st.rerun()
                with nav_cols[1]:
                    if st.button("Save score and next chunk", type="primary", key=f"sim_save_{chunk_signature}"):
                        sim_row = {
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "session_id": st.session_state.get("simultaneous_session_id", ""),
                            "script_id": sim_script_id,
                            "title": sim_title,
                            "source_file": sim_name,
                            "practice_mode": sim_mode,
                            "training_mode": sim_training_mode,
                            "lag_seconds": sim_lag_seconds if sim_training_mode in {"Lag trainer", "Shadow first, then interpret"} else "",
                            "playback_speed": sim_playback_speed,
                            "repeat_loop": sim_repeat_loop,
                            "chunk_index": chunk.get("chunk_index", sim_idx + 1),
                            "chunk_label": chunk.get("chunk_label", ""),
                            "source_language": source_lang,
                            "target_language": target_lang,
                            "speaker": chunk.get("speaker", ""),
                            "segment_type": chunk.get("segment_type", ""),
                            "score": sim_score,
                            "error_tags": "; ".join(sim_tags),
                            "notes": sim_notes,
                            "source_text": source_text,
                            "target_text": target_text,
                        }
                        st.session_state.simultaneous_log.append(sim_row)
                        if auto_save_simultaneous_log:
                            try:
                                append_simultaneous_log(simultaneous_log_path, sim_row)
                            except Exception as exc:
                                st.warning(f"Could not auto-save simultaneous practice: {exc}")
                        if sim_idx < len(chunks) - 1:
                            st.session_state.simultaneous_idx += 1
                            st.session_state.simultaneous_revealed = False
                        st.rerun()
                with nav_cols[2]:
                    if st.button("Skip chunk →", disabled=sim_idx >= len(chunks) - 1, key=f"sim_skip_{chunk_signature}"):
                        st.session_state.simultaneous_idx += 1
                        st.session_state.simultaneous_revealed = False
                        st.rerun()

            if not st.session_state.simultaneous_revealed:
                nav1, nav2 = st.columns(2)
                with nav1:
                    if st.button("← Previous", disabled=sim_idx == 0, key=f"sim_prev_unrevealed_{chunk_signature}"):
                        st.session_state.simultaneous_idx -= 1
                        st.session_state.simultaneous_revealed = False
                        st.rerun()
                with nav2:
                    if st.button("Skip →", disabled=sim_idx >= len(chunks) - 1, key=f"sim_skip_unrevealed_{chunk_signature}"):
                        st.session_state.simultaneous_idx += 1
                        st.session_state.simultaneous_revealed = False
                        st.rerun()

            st.divider()
            st.markdown("### Simultaneous session log")
            sim_session_rows = st.session_state.get("simultaneous_log", [])
            if sim_session_rows:
                st.dataframe(sim_session_rows, use_container_width=True, hide_index=True)
                sim_scores = [int(r.get("score", 0)) for r in sim_session_rows if str(r.get("score", "")).isdigit()]
                if sim_scores:
                    st.write(f"Average simultaneous score: **{sum(sim_scores) / len(sim_scores):.2f} / 3** across **{len(sim_scores)} saved attempt(s)**.")
                if auto_save_simultaneous_log:
                    st.caption(f"Simultaneous attempts are being saved to: `{simultaneous_log_path}`")
                st.download_button(
                    "Download simultaneous session CSV",
                    data=simultaneous_log_to_csv(sim_session_rows).encode("utf-8"),
                    file_name=f"{sim_script_id}_simultaneous_session_log.csv",
                    mime="text/csv",
                )
                if st.button("Clear simultaneous session log", key="sim_clear_session"):
                    st.session_state.simultaneous_log = []
                    st.rerun()
            else:
                st.info("No simultaneous attempts saved in this app session yet.")

            with st.expander("Persistent simultaneous history", expanded=False):
                sim_history_rows = load_simultaneous_log(simultaneous_log_path)
                sim_history_df = simultaneous_rows_to_dataframe(sim_history_rows)
                st.write(f"Persistent simultaneous log file: `{simultaneous_log_path}`")
                if sim_history_df.empty:
                    st.info("No persistent simultaneous practice history yet.")
                else:
                    st.metric("Saved simultaneous attempts", len(sim_history_df))
                    valid_sim_scores = sim_history_df["score_numeric"].dropna()
                    if not valid_sim_scores.empty:
                        st.metric("Average saved simultaneous score", f"{valid_sim_scores.mean():.2f} / 3")
                    st.dataframe(sim_history_df.tail(int(show_history_preview_rows)), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download full simultaneous history CSV",
                        data=simultaneous_log_to_csv(sim_history_rows).encode("utf-8"),
                        file_name="simultaneous_practice_log.csv",
                        mime="text/csv",
                    )
    else:
        st.info("Choose a JSON file to start simultaneous practice.")


with tab_history:
    st.subheader("Study history")
    st.caption("Phase Web 1.2: synced UI cleanup, side-by-side practice views, compact metadata, and consecutive/simultaneous playback speed controls.")
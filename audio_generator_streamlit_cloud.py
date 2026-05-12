from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import io
import json
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

import Current_Audio_Generator_Script_JSON_to_MP3_V7 as audio_gen
import generate_practice_docx

APP_TITLE = "Interpreter Audio Generator — Cloud"
RUNTIME_ROOT = Path(tempfile.gettempdir()) / "interpreter_audio_generator_cloud"
UPLOAD_DIR = RUNTIME_ROOT / "uploads"
AUDIO_OUT_DIR = RUNTIME_ROOT / "audio_output"
DOCX_OUT_DIR = RUNTIME_ROOT / "docx_output"
PRACTICE_CACHE_DIR = RUNTIME_ROOT / "practice_audio_cache"
for folder in [UPLOAD_DIR, AUDIO_OUT_DIR, DOCX_OUT_DIR, PRACTICE_CACHE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

ERROR_TAGS = [
    "Omission",
    "Number/date/time",
    "Wrong legal term",
    "Wrong speaker/actor",
    "Register/style",
    "Sequence/chronology",
    "Source-language interference",
]


def safe_filename(name: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
    cleaned = "".join(ch for ch in name if ch in keep).strip().replace(" ", "_")
    return cleaned or "uploaded_script.json"


def load_json_bytes(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def save_uploaded_files(uploaded_files) -> list[Path]:
    saved: list[Path] = []
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for uploaded in uploaded_files or []:
        raw = uploaded.getvalue()
        # Prefix with a short hash so two uploaded files with the same name do not collide.
        digest = hashlib.sha1(raw).hexdigest()[:10]
        path = UPLOAD_DIR / f"{digest}_{safe_filename(uploaded.name)}"
        path.write_bytes(raw)
        saved.append(path)
    return saved


def is_valid_script(data: dict[str, Any]) -> bool:
    return isinstance(data, dict) and bool(data.get("script_id")) and isinstance(data.get("paired_segments"), list)


def summarize_script(data: dict[str, Any]) -> dict[str, Any]:
    segments = data.get("paired_segments", []) or []
    speakers = sorted({str(seg.get("speaker", "")).strip() for seg in segments if str(seg.get("speaker", "")).strip()})
    types = sorted({str(seg.get("segment_type", "")).strip() for seg in segments if str(seg.get("segment_type", "")).strip()})
    en_source = sum(1 for seg in segments if str(seg.get("source_language", "")).lower() == "english")
    es_source = sum(1 for seg in segments if str(seg.get("source_language", "")).lower() == "spanish")
    mismatches = []
    for i, seg in enumerate(segments, start=1):
        src = str(seg.get("source_language", "")).lower().strip()
        order = seg.get("playback_order") or []
        first = str(order[0]).lower().strip() if order else ""
        if src and first and src != first:
            mismatches.append(i)
    return {
        "Script ID": data.get("script_id", ""),
        "Title": data.get("title", ""),
        "Format": data.get("format", ""),
        "Segments": len(segments),
        "Terms": len(data.get("terms_used", []) or []),
        "English-source": en_source,
        "Spanish-source": es_source,
        "Speakers": ", ".join(speakers[:12]) + ("…" if len(speakers) > 12 else ""),
        "Segment types": ", ".join(types[:12]) + ("…" if len(types) > 12 else ""),
        "Playback mismatches": len(mismatches),
    }


def make_zip_bytes(paths: list[Path]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in paths:
            if path.exists() and path.is_file():
                zf.write(path, arcname=path.name)
    bio.seek(0)
    return bio.getvalue()


def configure_audio_generator(settings: dict[str, Any]) -> None:
    AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
    audio_gen.OUTPUT_DIR = AUDIO_OUT_DIR
    audio_gen.SPEED_MODE = settings["speed_mode"]
    audio_gen.INCLUDE_FLASHCARDS = settings["include_flashcards"]
    audio_gen.SPLIT_FLASHCARDS_AND_SCRIPT = settings["split_outputs"]
    audio_gen.GENERATE_CONSECUTIVE_MODE = settings["consecutive_mode"]
    audio_gen.CONSECUTIVE_PAUSE_RATIO = settings["consecutive_ratio"]
    audio_gen.CONSECUTIVE_MIN_PAUSE = settings["consecutive_min"]
    audio_gen.CONSECUTIVE_MAX_PAUSE = settings["consecutive_max"]
    audio_gen.CONSECUTIVE_SEGMENT_GAP = settings["consecutive_gap"]
    audio_gen.GENERATE_FULL_SPEED_INCREASE = settings["full_speed"]
    audio_gen.GENERATE_TARGETED_HARD_TERMS = settings["targeted_terms"]
    audio_gen.GENERATE_INLINE_TERM_CUES = settings["inline_cues"]
    audio_gen.GENERATE_COMBINED_AUDIO_ALL_SCRIPTS = False
    audio_gen.SKIP_EXISTING_SCRIPT_MP3S = False
    audio_gen.FORCE_REBUILD_COMBINED = True
    audio_gen.PRINT_VOICE_ASSIGNMENTS = False
    audio_gen.VERIFY_TTS_VOICES = settings["verify_voices"]
    audio_gen.USE_SPEAKER_VOICE_ASSIGNMENTS = settings["speaker_voices"]


async def generate_audio_for_paths(paths: list[Path]) -> list[Path]:
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg was not found. In Streamlit Cloud, add ffmpeg to packages.txt.")
    await audio_gen.initialize_voice_validation()
    outputs: list[Path] = []
    for path in paths:
        outputs.extend(await audio_gen.build_audio_for_script(path))
    return outputs


def get_segment_languages(seg: dict[str, Any]) -> tuple[str, str]:
    order = audio_gen._normalize_playback_order(seg.get("playback_order")) or []
    if len(order) >= 2:
        return order[0], order[1]
    src = str(seg.get("source_language", "english")).lower().strip() or "english"
    tgt = "spanish" if src == "english" else "english"
    return src, tgt


def text_for_language(data: dict[str, Any], seg: dict[str, Any], lang: str) -> str:
    try:
        return audio_gen.text_for_tts(data, seg, lang)
    except Exception:
        return str(seg.get(lang, ""))


def practice_audio_path(data: dict[str, Any], seg: dict[str, Any], lang: str, role: str, speed_mode: str) -> Path:
    script_id = str(data.get("script_id", "script"))
    idx = int(seg.get("_practice_index", 0))
    text = text_for_language(data, seg, lang)
    voice = audio_gen.get_voice_for_segment(data, seg, lang, audio_gen.EN_VOICE if lang == "english" else audio_gen.ES_VOICE)
    digest = hashlib.sha1(f"{script_id}|{idx}|{role}|{lang}|{voice}|{speed_mode}|{text}".encode("utf-8")).hexdigest()[:16]
    return PRACTICE_CACHE_DIR / f"{safe_filename(script_id)}_{idx:04d}_{role}_{lang}_{digest}.mp3"


async def ensure_practice_audio(data: dict[str, Any], seg: dict[str, Any], lang: str, role: str, speed_mode: str) -> Path:
    path = practice_audio_path(data, seg, lang, role, speed_mode)
    if path.exists() and path.stat().st_size > 0:
        return path
    speed = audio_gen.SPEED_PRESETS.get(speed_mode, audio_gen.SPEED_PRESETS["learning"])
    rate = speed["en_rate"] if lang == "english" else speed["es_rate"]
    voice = audio_gen.get_voice_for_segment(data, seg, lang, audio_gen.EN_VOICE if lang == "english" else audio_gen.ES_VOICE)
    text = text_for_language(data, seg, lang)
    await audio_gen.initialize_voice_validation()
    await audio_gen.tts_to_mp3(text, voice, rate, path)
    return path


def audio_html(path: Path, autoplay: bool, key_seed: str) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    auto = "autoplay" if autoplay else ""
    return f"""
    <audio id="audio-{key_seed}" controls {auto} style="width:100%;">
      <source src="data:audio/mp3;base64,{b64}" type="audio/mpeg">
    </audio>
    """


def current_session_rows() -> list[dict[str, Any]]:
    return st.session_state.setdefault("practice_rows", [])


def reset_practice_state() -> None:
    st.session_state.practice_order = []
    st.session_state.practice_pos = 0
    st.session_state.practice_revealed = False
    st.session_state.practice_rows = []
    st.session_state.source_visible = False


def detect_terms_in_segment(data: dict[str, Any], source_text: str, target_text: str) -> list[str]:
    hay = f" {source_text} {target_text} ".lower()
    hits = []
    for term in data.get("terms_used", []) or []:
        english = str(term.get("english", "")).strip()
        spanish = str(term.get("spanish_contextual_choice", "")).strip()
        alts = [str(x).strip() for x in term.get("spanish_alternatives_from_sheet", []) or []]
        labels = [english, spanish] + alts
        if any(label and label.lower() in hay for label in labels):
            hits.append(f"{english} → {spanish}" if english and spanish else english or spanish)
    return hits[:12]


st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Cloud-safe version: upload JSON files, generate downloadable outputs, and practice in-session.")

uploaded_files = st.sidebar.file_uploader(
    "Upload script JSON file(s)",
    type=["json"],
    accept_multiple_files=True,
    help="Uploaded files stay in this running session. Download generated outputs before closing the app.",
)

saved_paths = save_uploaded_files(uploaded_files)
loaded_scripts: list[tuple[Path, dict[str, Any]]] = []
for path in saved_paths:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if is_valid_script(data):
            for i, seg in enumerate(data.get("paired_segments", []) or [], start=1):
                seg["_practice_index"] = i
            loaded_scripts.append((path, data))
    except Exception as exc:
        st.sidebar.error(f"Could not read {path.name}: {exc}")

if not loaded_scripts:
    st.info("Upload one or more compatible JSON scripts to begin.")
    st.stop()

script_labels = [f"{data.get('script_id', path.stem)} — {data.get('title', '')[:55]}" for path, data in loaded_scripts]

tab_preview, tab_generate, tab_practice, tab_help = st.tabs([
    "Preview / Validate",
    "Generate outputs",
    "Consecutive practice",
    "Deployment notes",
])

with tab_preview:
    st.subheader("Uploaded script preview")
    rows = [summarize_script(data) for _, data in loaded_scripts]
    st.dataframe(rows, use_container_width=True)
    for path, data in loaded_scripts:
        with st.expander(f"{data.get('script_id', path.stem)} details"):
            st.json({
                "script_id": data.get("script_id"),
                "title": data.get("title"),
                "format": data.get("format"),
                "category": data.get("category"),
                "subcategory": data.get("subcategory"),
                "audio_profile_present": bool(data.get("audio_profile") or data.get("voice_assignments")),
            })
            mismatches = []
            for i, seg in enumerate(data.get("paired_segments", []) or [], start=1):
                src = str(seg.get("source_language", "")).lower().strip()
                order = seg.get("playback_order") or []
                first = str(order[0]).lower().strip() if order else ""
                if src and first and src != first:
                    mismatches.append((i, src, order))
            if mismatches:
                st.warning("Playback/source-language mismatches found.")
                st.dataframe([{"segment": i, "source_language": s, "playback_order": o} for i, s, o in mismatches])
            else:
                st.success("No source-language/playback-order mismatches found.")

with tab_generate:
    st.subheader("Generate downloadable audio and practice packets")
    st.caption("Cloud mode writes outputs to temporary server storage and provides download buttons.")

    with st.expander("Audio settings", expanded=True):
        c1, c2, c3 = st.columns(3)
        speed_mode = c1.selectbox("Speed mode", list(audio_gen.SPEED_PRESETS.keys()), index=0)
        split_outputs = c2.checkbox("Split flashcards and script", value=True)
        include_flashcards = c3.checkbox("Include flashcards", value=True)
        c4, c5, c6 = st.columns(3)
        consecutive_mode = c4.checkbox("Consecutive mode", value=True)
        full_speed = c5.checkbox("Full speed-increase drill", value=False)
        targeted_terms = c6.checkbox("Targeted hard-terms drill", value=False)
        c7, c8, c9 = st.columns(3)
        inline_cues = c7.checkbox("Inline term cues", value=False)
        speaker_voices = c8.checkbox("Use JSON speaker voices", value=True)
        verify_voices = c9.checkbox("Verify Edge TTS voices", value=True)
        st.markdown("**Consecutive mode timing**")
        t1, t2, t3, t4 = st.columns(4)
        consecutive_ratio = t1.number_input("Pause ratio", value=1.35, min_value=0.2, max_value=4.0, step=0.05)
        consecutive_min = t2.number_input("Minimum pause", value=3.0, min_value=0.0, max_value=30.0, step=0.5)
        consecutive_max = t3.number_input("Maximum pause", value=14.0, min_value=1.0, max_value=60.0, step=0.5)
        consecutive_gap = t4.number_input("Segment gap", value=1.8, min_value=0.0, max_value=10.0, step=0.1)

    with st.expander("Practice packet settings", expanded=False):
        p1, p2, p3 = st.columns(3)
        gen_docx = p1.checkbox("Individual DOCX", value=True)
        gen_pdf = p2.checkbox("Individual PDF", value=False)
        highlight_terms = p3.checkbox("Highlight glossary terms", value=True)
        p4, p5 = st.columns(2)
        gen_master_docx = p4.checkbox("Combined master DOCX", value=False)
        gen_master_pdf = p5.checkbox("Combined master PDF", value=False)
        master_name = st.text_input("Combined packet filename", value="combined_practice_packet.docx")

    settings = {
        "speed_mode": speed_mode,
        "include_flashcards": include_flashcards,
        "split_outputs": split_outputs,
        "consecutive_mode": consecutive_mode,
        "consecutive_ratio": consecutive_ratio,
        "consecutive_min": consecutive_min,
        "consecutive_max": consecutive_max,
        "consecutive_gap": consecutive_gap,
        "full_speed": full_speed,
        "targeted_terms": targeted_terms,
        "inline_cues": inline_cues,
        "speaker_voices": speaker_voices,
        "verify_voices": verify_voices,
    }

    b1, b2 = st.columns(2)
    if b1.button("Generate audio", type="primary"):
        AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for old in AUDIO_OUT_DIR.glob("*"):
            if old.is_file():
                old.unlink()
        configure_audio_generator(settings)
        try:
            with st.spinner("Generating audio. This may take a while for large scripts."):
                outputs = asyncio.run(generate_audio_for_paths([p for p, _ in loaded_scripts]))
            st.success(f"Generated {len(outputs)} audio file(s).")
            if outputs:
                zip_bytes = make_zip_bytes(outputs)
                st.download_button("Download audio ZIP", zip_bytes, file_name="interpreter_audio_outputs.zip", mime="application/zip")
                st.dataframe([{"file": p.name, "size_mb": round(p.stat().st_size / 1_000_000, 2)} for p in outputs])
        except Exception as exc:
            st.exception(exc)

    if b2.button("Generate practice packet(s)"):
        DOCX_OUT_DIR.mkdir(parents=True, exist_ok=True)
        for old in DOCX_OUT_DIR.glob("*"):
            if old.is_file():
                old.unlink()
        try:
            with st.spinner("Generating practice packet files."):
                outputs = generate_practice_docx.generate_practice_packets(
                    [p for p, _ in loaded_scripts],
                    DOCX_OUT_DIR,
                    generate_individual_docx=gen_docx,
                    generate_individual_pdfs=gen_pdf,
                    generate_master_docx=gen_master_docx,
                    generate_master_pdf=gen_master_pdf,
                    highlight_terms=highlight_terms,
                    master_docx_name=master_name,
                )
            st.success(f"Generated {len(outputs)} practice packet file(s).")
            if outputs:
                zip_bytes = make_zip_bytes(outputs)
                st.download_button("Download practice packet ZIP", zip_bytes, file_name="interpreter_practice_packets.zip", mime="application/zip")
                st.dataframe([{"file": p.name, "size_mb": round(p.stat().st_size / 1_000_000, 2)} for p in outputs])
        except Exception as exc:
            st.exception(exc)

with tab_practice:
    st.subheader("Interactive consecutive practice")
    st.caption("Current-session practice only. Download your session log before closing the app.")

    selected_idx = st.selectbox("Choose script", range(len(loaded_scripts)), format_func=lambda i: script_labels[i])
    _, practice_data = loaded_scripts[selected_idx]
    all_segments = practice_data.get("paired_segments", []) or []

    setup_col, progress_col = st.columns([2, 1])
    with setup_col:
        mode = st.selectbox("Practice mode", ["Consecutive Standard", "Consecutive Beginner", "Consecutive Exam Mode", "Review Missed Segments"])
        audio_only_source = st.checkbox("Audio-only source prompt", value=(mode == "Consecutive Exam Mode"))
        randomize = st.checkbox("Randomize segments", value=False)
        segment_types = sorted({str(s.get("segment_type", "")).strip() for s in all_segments if str(s.get("segment_type", "")).strip()})
        selected_types = st.multiselect("Segment types", segment_types, default=segment_types)
    with progress_col:
        st.metric("Session rows", len(current_session_rows()))
        avg = sum(float(r.get("score", 0)) for r in current_session_rows()) / len(current_session_rows()) if current_session_rows() else 0
        st.metric("Average score", f"{avg:.2f}/3" if current_session_rows() else "—")

    with st.expander("Advanced practice audio settings", expanded=False):
        pa1, pa2, pa3 = st.columns(3)
        practice_audio = pa1.checkbox("Enable practice audio", value=True)
        autoplay_source = pa2.checkbox("Auto-play source audio", value=True)
        play_correction_audio = pa3.checkbox("Play correction audio on reveal", value=True)
        use_speaker_voices_practice = st.checkbox("Use JSON audio_profile speaker voices", value=True)
        if st.button("Clear practice audio cache"):
            shutil.rmtree(PRACTICE_CACHE_DIR, ignore_errors=True)
            PRACTICE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            st.success("Practice audio cache cleared.")

    audio_gen.USE_SPEAKER_VOICE_ASSIGNMENTS = use_speaker_voices_practice
    audio_gen.VERIFY_TTS_VOICES = True

    filtered = [s for s in all_segments if not selected_types or str(s.get("segment_type", "")).strip() in selected_types]
    if mode == "Review Missed Segments" and current_session_rows():
        weak_indexes = {int(r["segment_index"]) for r in current_session_rows() if float(r.get("score", 3)) < 3}
        filtered = [s for s in filtered if int(s.get("_practice_index", 0)) in weak_indexes]

    if st.button("Start / Reset practice") or "practice_order" not in st.session_state:
        reset_practice_state()
        order = list(range(len(filtered)))
        if randomize:
            import random
            random.shuffle(order)
        st.session_state.practice_order = order
        st.session_state.practice_pos = 0

    if not filtered:
        st.warning("No segments match the current filters.")
        st.stop()

    order = st.session_state.get("practice_order") or list(range(len(filtered)))
    pos = min(int(st.session_state.get("practice_pos", 0)), max(len(order) - 1, 0))
    st.session_state.practice_pos = pos
    seg = filtered[order[pos]]
    source_lang, target_lang = get_segment_languages(seg)
    source_text = text_for_language(practice_data, seg, source_lang)
    target_text = text_for_language(practice_data, seg, target_lang)

    st.progress((pos + 1) / max(len(order), 1), text=f"Segment {pos + 1} of {len(order)}")

    st.markdown(
        f"""
        <div style='padding:1rem;border:1px solid #ddd;border-radius:14px;margin:0.5rem 0;background:#fafafa;'>
        <strong>Speaker:</strong> {seg.get('speaker', '—')} &nbsp; | &nbsp;
        <strong>Direction:</strong> {source_lang.title()} → {target_lang.title()} &nbsp; | &nbsp;
        <strong>Type:</strong> {seg.get('segment_type', '—')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if practice_audio:
        try:
            source_path = asyncio.run(ensure_practice_audio(practice_data, seg, source_lang, "source", speed_mode))
            seed = f"source-{practice_data.get('script_id')}-{seg.get('_practice_index')}-{source_path.stat().st_mtime_ns}"
            st.markdown("**▶ Source audio**", unsafe_allow_html=False)
            st.components.v1.html(audio_html(source_path, autoplay_source, seed), height=62)
            if st.button("↻ Replay source", key=f"replay_source_{seg.get('_practice_index')}_{time.time_ns()}"):
                st.components.v1.html(audio_html(source_path, True, seed + "-replay"), height=62)
        except Exception as exc:
            st.error(f"Could not create source audio: {exc}")

    if audio_only_source and not st.session_state.get("source_visible", False):
        if st.button("Show source text"):
            st.session_state.source_visible = True
            st.rerun()
    else:
        st.markdown("### Original / Source")
        st.info(source_text)

    if st.button("👁 Reveal correction", type="primary"):
        st.session_state.practice_revealed = True

    if st.session_state.get("practice_revealed", False):
        st.markdown("### Correction / Target")
        st.success(target_text)
        if practice_audio and play_correction_audio:
            try:
                target_path = asyncio.run(ensure_practice_audio(practice_data, seg, target_lang, "target", speed_mode))
                seed = f"target-{practice_data.get('script_id')}-{seg.get('_practice_index')}-{target_path.stat().st_mtime_ns}"
                st.markdown("**▶ Correction audio**", unsafe_allow_html=False)
                st.components.v1.html(audio_html(target_path, True, seed), height=62)
            except Exception as exc:
                st.error(f"Could not create correction audio: {exc}")

        terms = detect_terms_in_segment(practice_data, source_text, target_text)
        if terms:
            with st.expander("Terms in this segment", expanded=True):
                for item in terms:
                    st.write(f"• {item}")

        score = st.radio("Self-score", [3, 2, 1, 0], horizontal=True, format_func=lambda x: {3:"3 Accurate",2:"2 Minor issue",1:"1 Major issue",0:"0 Missed"}[x])
        tags = st.multiselect("Error tags", ERROR_TAGS)
        notes = st.text_area("Notes", height=80)

        nav1, nav2, nav3 = st.columns(3)
        if nav1.button("✅ Save + Next"):
            current_session_rows().append({
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "script_id": practice_data.get("script_id"),
                "segment_index": seg.get("_practice_index"),
                "speaker": seg.get("speaker", ""),
                "segment_type": seg.get("segment_type", ""),
                "direction": f"{source_lang}->{target_lang}",
                "score": score,
                "error_tags": "; ".join(tags),
                "notes": notes,
                "source_text": source_text,
                "target_text": target_text,
            })
            st.session_state.practice_pos = min(pos + 1, len(order) - 1)
            st.session_state.practice_revealed = False
            st.session_state.source_visible = False
            st.rerun()
        if nav2.button("⬅ Previous"):
            st.session_state.practice_pos = max(pos - 1, 0)
            st.session_state.practice_revealed = False
            st.session_state.source_visible = False
            st.rerun()
        if nav3.button("➡ Skip"):
            st.session_state.practice_pos = min(pos + 1, len(order) - 1)
            st.session_state.practice_revealed = False
            st.session_state.source_visible = False
            st.rerun()

    if current_session_rows():
        with st.expander("Session log", expanded=False):
            st.dataframe(current_session_rows(), use_container_width=True)
            csv_bio = io.StringIO()
            writer = csv.DictWriter(csv_bio, fieldnames=list(current_session_rows()[0].keys()))
            writer.writeheader()
            writer.writerows(current_session_rows())
            st.download_button("Download session log CSV", csv_bio.getvalue(), file_name="practice_session_log.csv", mime="text/csv")

with tab_help:
    st.subheader("Cloud deployment notes")
    st.markdown(
        """
        This cloud version is designed for Streamlit Community Cloud:

        - Upload JSON files with the sidebar uploader.
        - Generated MP3/DOCX/PDF files are temporary and should be downloaded as ZIP files.
        - Practice history is session-only; download the CSV before closing the browser tab.
        - `packages.txt` installs `ffmpeg` on Streamlit Cloud.
        - Keep large/generated outputs out of GitHub.
        """
    )

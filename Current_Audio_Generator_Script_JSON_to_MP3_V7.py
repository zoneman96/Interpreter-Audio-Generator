import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import edge_tts

SCRIPTS_DIR = Path("scripts")
OUTPUT_DIR = Path("audio_output")
OUTPUT_DIR.mkdir(exist_ok=True)

EN_VOICE = "en-US-JennyNeural"
ES_VOICE = "es-MX-DaliaNeural"

# -----------------------------
# Optional speaker/participant voice support
# -----------------------------
# Backwards compatible: older JSON files with no audio_profile/voice_assignments
# still use EN_VOICE and ES_VOICE exactly as before.
#
# New JSONs may include either a top-level voice_assignments block:
#   "voice_assignments": {"PARENT": {"english": "en-US-JennyNeural", "spanish": "es-MX-DaliaNeural"}}
#
# or the preferred audio_profile block:
#   "audio_profile": {
#       "voice_mode": "speaker",
#       "default_voices": {"english": "en-US-JennyNeural", "spanish": "es-MX-DaliaNeural"},
#       "voice_assignments": {"PARENT": {"english": "en-US-JennyNeural", "spanish": "es-MX-DaliaNeural"}}
#   }
#
# If INTERPRETED_AUDIO_USES_INTERPRETER_VOICE is False, both the source utterance
# and the interpretation use the segment speaker's assigned voice. This is usually
# best for speaker-discrimination drills. If True, interpreted chunks use the
# INTERPRETER assignment when available.
USE_SPEAKER_VOICE_ASSIGNMENTS = True
INTERPRETED_AUDIO_USES_INTERPRETER_VOICE = False
# Keep speaker labels in JSON for structure/voice assignment, but do not speak
# labels such as "JUDGE:" or "PARENT:" in generated audio by default.
# Per-script override: audio_profile.speak_speaker_labels = true/false.
SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO = os.environ.get("SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO", "0") != "0"
INTERPRETER_SPEAKER_KEYS = {"INTERPRETER", "INTÉRPRETE", "INTERPRETE"}

# Handy reference only. The script does not require these exact voices; any valid
# edge-tts voice name may be used in the JSON.
VOICE_LIBRARY = {
    "english_default_female": "en-US-JennyNeural",
    "english_default_male": "en-US-GuyNeural",
    "english_neutral_female": "en-US-AriaNeural",
    "english_younger_male": "en-US-RogerNeural",
    "english_adult_male_alt": "en-US-DavisNeural",
    "spanish_default_female": "es-MX-DaliaNeural",
    "spanish_default_male": "es-MX-JorgeNeural",
}

# -----------------------------
# Recommended settings
# -----------------------------
# Default profile: Learning
# Other profiles you can switch to quickly:
#
# Integration profile:
#   SPEED_MODE = "normal"
#   TARGETED_INTERPRET_PAUSE = 4.0
#   TARGETED_RETRY_PAUSE = 3.6
#   INLINE_SENTENCE_PAUSE = 1.2
#
# Pressure profile:
#   SPEED_MODE = "fast"
#   TARGETED_INTERPRET_PAUSE = 3.2
#   TARGETED_RETRY_PAUSE = 3.0
#   INLINE_SENTENCE_PAUSE = 1.0

SPEED_MODE = "learning"
SPEED_PRESETS = {
    "learning": {"en_rate": "-17%", "es_rate": "-17%"},
    "normal": {"en_rate": "-4%", "es_rate": "-4%"},
    "fast": {"en_rate": "+6%", "es_rate": "+6%"},
}

FLASHCARD_EN_ES_PAUSE = 1.1
FLASHCARD_ALT_PAUSE = 0.8
FLASHCARD_TERM_GAP = 1.4
SEGMENT_EN_ES_PAUSE = 1.0
SEGMENT_GAP = 1.6
SECTION_GAP = 2.0
COMBINED_SCRIPT_GAP = 3.0

READ_ALL_SPANISH_OPTIONS = True
INCLUDE_FLASHCARDS = True
SPLIT_FLASHCARDS_AND_SCRIPT = True
PREFER_PAIRED_SEGMENTS = True
MP3_BITRATE = "64k"
SKIP_EXISTING_SCRIPT_MP3S = os.environ.get("SKIP_EXISTING_SCRIPT_MP3S", "1") != "0"
FORCE_REBUILD_COMBINED = os.environ.get("FORCE_REBUILD_COMBINED", "1") != "0"
GENERATE_COMBINED_AUDIO_ALL_SCRIPTS = False

# Rebuild behavior for speaker-voice testing and JSON edits.
# Older scripts can still be skipped when outputs exist. JSONs with audio_profile
# rebuild by default so new/changed voice assignments are actually applied.
FORCE_REBUILD_WHEN_AUDIO_PROFILE_PRESENT = os.environ.get("FORCE_REBUILD_WHEN_AUDIO_PROFILE_PRESENT", "0") != "0"
REBUILD_IF_JSON_NEWER_THAN_OUTPUT = os.environ.get("REBUILD_IF_JSON_NEWER_THAN_OUTPUT", "1") != "0"
PRINT_VOICE_ASSIGNMENTS = os.environ.get("PRINT_VOICE_ASSIGNMENTS", "1") != "0"

# Voice validation/fallback. This prevents edge_tts.exceptions.NoAudioReceived
# when a JSON names a voice that is unavailable in your installed Edge TTS catalog.
VERIFY_TTS_VOICES = os.environ.get("VERIFY_TTS_VOICES", "1") != "0"
VALID_EDGE_VOICES: set[str] | None = None
WARNED_INVALID_VOICES: set[str] = set()

# Targeted hard-terms mode
GENERATE_TARGETED_HARD_TERMS = False
TARGETED_INTERPRET_PAUSE = 4.8
TARGETED_TERM_PAUSE = 0.9
TARGETED_RETRY_PAUSE = 4.0
TARGETED_FINAL_FAST_PAUSE = 2.8

# Full speed-increase drill (separate optional output)
GENERATE_FULL_SPEED_INCREASE = True
FULL_SPEED_INCLUDE_FAST_SPANISH = False
FULL_SPEED_RATE_LEARNING = "-20%"
FULL_SPEED_RATE_NORMAL = "-10%"
FULL_SPEED_RATE_FAST = "+1%"
FULL_SPEED_INTERPRET_PAUSE_LEARNING = 4.8
FULL_SPEED_INTERPRET_PAUSE_NORMAL = 3.8
FULL_SPEED_FINAL_FAST_PAUSE = 2.8
FULL_SPEED_SHORT_PAUSE = 0.9
FULL_SPEED_SEGMENT_GAP = 1.8

# Consecutive interpreting practice mode (separate optional output)
# Output pattern: <script_id>_consecutive_mode.mp3
# Each paired segment plays the original/source language first, inserts a
# dynamic pause based on the actual source audio duration, then plays the
# interpreted/translated language for correction. No dynamic pause is added
# after the interpreted/translated language.
GENERATE_CONSECUTIVE_MODE = True
CONSECUTIVE_PAUSE_RATIO = 1.35
CONSECUTIVE_MIN_PAUSE = 3.0
CONSECUTIVE_MAX_PAUSE = 14.0
CONSECUTIVE_SEGMENT_GAP = 1.8

# Source-only practice mode (separate optional output)
# Output pattern: <script_id>_source_only.mp3
# Plays only the original/source side of each paired segment while preserving
# JSON speaker voice assignments. Intended for real-time interpreting practice
# without the included translation/correction.
GENERATE_SOURCE_ONLY_AUDIO = True
# If True, adjacent source-language JSON segments with the same speaker, same
# source language, and same assigned TTS voice are combined into one TTS request
# for more natural source-only practice audio. This avoids artificial pauses caused
# only by JSON chunking while still preserving pauses when the speaker changes.
SOURCE_ONLY_MERGE_ADJACENT_SAME_SPEAKER = True
# Safety limit for a merged same-speaker TTS block. If a long monologue exceeds
# this size, the generator starts a new same-speaker block and uses the
# continuation gap below.
SOURCE_ONLY_MAX_MERGED_CHARS = 2800
SOURCE_ONLY_SEGMENT_GAP = 0.75
SOURCE_ONLY_SPEAKER_CHANGE_GAP = 1.35
SOURCE_ONLY_MERGED_CONTINUATION_GAP = 0.18

# Retry transient Edge TTS failures
TTS_MAX_RETRIES = 5
TTS_RETRY_BASE_SECONDS = 2.0
TTS_RETRY_MAX_SECONDS = 20.0
TARGETED_SEGMENT_GAP = 1.8
TARGETED_INCLUDE_FULL_SPANISH_AT_END = True
TARGETED_REPEAT_ENGLISH_CHUNK = True
TARGETED_INCLUDE_FAST_FINAL_RECALL = True
TARGETED_MATCH_TERMS_IN_SEGMENTS = True
TARGETED_MIN_TERM_LENGTH = 4
TARGETED_RATE_LEARNING = "-17%"
TARGETED_RATE_NORMAL = "-4%"
TARGETED_RATE_FAST = "+6%"

# Inline term cue mode
GENERATE_INLINE_TERM_CUES = False
INLINE_SENTENCE_PAUSE = 1.5
INLINE_CUE_PAUSE = 0.8
INLINE_SEGMENT_GAP = 1.6
# Recommended: keep False so cue placement is anchored to the English sentence only.
INLINE_REQUIRE_MATCH_IN_ENGLISH_OR_SPANISH = False


def get_speed_config() -> dict[str, str]:
    if SPEED_MODE not in SPEED_PRESETS:
        raise ValueError(f"Unknown SPEED_MODE={SPEED_MODE!r}. Choose from {list(SPEED_PRESETS)}")
    return SPEED_PRESETS[SPEED_MODE]


def get_target_names() -> set[str] | None:
    raw = os.environ.get("PIPELINE_TARGETS")
    if not raw:
        return None
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return {str(x) for x in items}
    except json.JSONDecodeError:
        pass
    return None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_flashcard_tts_text(text: str, language: str) -> str:
    """Normalize glossary/flashcard text for natural TTS without changing JSON data.

    Used for flashcards only. This keeps visual notation in JSON, such as
    "inquilino / arrendatario", while avoiding spoken output such as
    "barra diagonal".

    Optional JSON fields supported by build_flashcard_lines:
      - english_tts: overrides the spoken English flashcard term
      - spanish_tts: overrides the spoken Spanish contextual choice
    """
    spoken = normalize_space(text)
    if not spoken:
        return spoken

    lang = normalize_space(language).casefold()
    is_spanish = lang.startswith("spanish") or lang.startswith("es")
    separator_word = " o " if is_spanish else " or "

    # Slash as either/or separator. Avoids dates like 04/28/2026.
    spoken = re.sub(r"\s+/\s+", separator_word, spoken)
    letter_class = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ"
    spoken = re.sub(fr"(?<=[{letter_class}])/(?=[{letter_class}])", separator_word, spoken)

    if is_spanish:
        replacements = {
            "→": " se traduce como ",
            "⇒": " se traduce como ",
            "&": " y ",
            "=": " es igual a ",
            ";": ", ",
        }
    else:
        replacements = {
            "→": " translates as ",
            "⇒": " translates as ",
            "&": " and ",
            "=": " equals ",
            ";": ", ",
        }

    for old, new in replacements.items():
        spoken = spoken.replace(old, new)

    return normalize_space(spoken)


def normalize_no_abbreviation_tts_text(text: str) -> str:
    """Prevent TTS from reading answer-like "No." as "number".

    Keeps legitimate abbreviation uses where context clearly means "number",
    such as "Case No. CI26-9703" or "No. 12", while making short
    dialogue answers like "No." and sentence-start answers like
    "No. I did not receive it" sound natural.
    """
    spoken = str(text or "")
    if not spoken:
        return spoken

    # Clear legal/admin abbreviation contexts: "Case No. CI26-9703" ->
    # "Case Number CI26-9703". This avoids relying on Edge TTS to infer it.
    spoken = re.sub(
        r"(?i)\b(case|cause|matter|docket|file|item|exhibit|count|citation|ticket|apartment|apt|unit|room)\s+no\.?(?=\s+[A-Za-z0-9#-])",
        lambda m: f"{m.group(1)} Number",
        spoken,
    )

    # Standalone numeric abbreviation: "No. 12" -> "Number 12".
    # This only triggers when a digit follows, so "No. I did not" is not affected.
    spoken = re.sub(r"(?i)\bno\.\s+(?=\d)", "Number ", spoken)

    # Standalone dialogue answer: "No." -> "No".
    spoken = re.sub(r"^\s*no\.\s*$", "No", spoken, flags=re.IGNORECASE)

    # Sentence-start dialogue answer: "No. I did not..." -> "No, I did not...".
    spoken = re.sub(r"^\s*no\.\s+", "No, ", spoken, flags=re.IGNORECASE)

    # Occasional quoted or parenthetical answer-like forms: "He said, No." ->
    # "He said, No". Avoids touching "Case No." because those were handled above.
    spoken = re.sub(r"(?i)(?<=[\s\(\[\{\"'])no\.(?=\s*$|[\)\]\}\"'])", "No", spoken)

    return spoken

def clean_text(text: str) -> str:
    text = str(text or "")
    text = normalize_no_abbreviation_tts_text(text)
    text = text.replace("&", "and")
    text = text.replace("Q:", "Question. ")
    text = text.replace("A:", "Answer. ")
    text = text.replace("vs.", "versus")
    text = text.replace("v.", "versus")
    text = text.replace("Mr.", "Mister ")
    text = text.replace("Ms.", "Miss ")
    text = text.replace("Mrs.", "Misses ")
    text = text.replace("Dr.", "Doctor ")
    text = text.replace("Hon.", "Honorable ")
    text = text.replace("Dept.", "Department ")
    text = text.replace("Exh.", "Exhibit ")
    text = text.replace("Ex.", "Exhibit ")
    text = text.replace("U.S.", "U S")
    text = text.replace("D.A.", "D A")
    text = text.replace("P.O.", "P O")
    text = text.replace("TRO", "T R O")
    return normalize_space(text)


async def initialize_voice_validation() -> None:
    """Load the local Edge TTS voice catalog once, when available.

    If this lookup fails because of network/service issues, the generator still runs.
    In that case, tts_to_mp3 has a final fallback for NoAudioReceived errors.
    """
    global VALID_EDGE_VOICES
    if not VERIFY_TTS_VOICES:
        VALID_EDGE_VOICES = None
        return
    try:
        voices = await edge_tts.list_voices()
        VALID_EDGE_VOICES = {str(v.get("ShortName", "")).strip() for v in voices if v.get("ShortName")}
        print(f"Loaded {len(VALID_EDGE_VOICES)} Edge TTS voices for validation.")
    except Exception as exc:
        VALID_EDGE_VOICES = None
        print(f"Voice validation unavailable; continuing without pre-validation: {exc}")


def fallback_voice_for_name(voice: str) -> str:
    voice = str(voice or "")
    if voice.startswith("es-"):
        return ES_VOICE
    return EN_VOICE


def validate_voice_name(voice: str) -> str:
    voice = normalize_space(voice or "")
    if not voice:
        return EN_VOICE
    if VALID_EDGE_VOICES is not None and voice not in VALID_EDGE_VOICES:
        fallback = fallback_voice_for_name(voice)
        if voice not in WARNED_INVALID_VOICES:
            print(f"WARNING: Edge TTS voice '{voice}' is not available. Falling back to '{fallback}'.")
            WARNED_INVALID_VOICES.add(voice)
        return fallback
    return voice


async def tts_to_mp3(text: str, voice: str, rate: str, out_path: Path) -> None:
    cleaned = clean_text(text)
    requested_voice = normalize_space(voice or "")
    voice = validate_voice_name(requested_voice)
    fallback_voice = fallback_voice_for_name(requested_voice or voice)
    last_error = None
    tried_no_audio_fallback = False

    for attempt in range(1, TTS_MAX_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text=cleaned, voice=voice, rate=rate)
            await communicate.save(str(out_path))
            return
        except Exception as exc:
            last_error = exc
            message = str(exc)
            no_audio = "No audio was received" in message or exc.__class__.__name__ == "NoAudioReceived"

            # If the service returns no audio for a non-default assigned voice, retry once
            # with the language default. This preserves backwards compatibility and avoids
            # a full crash from one unavailable/retired voice name.
            if no_audio and not tried_no_audio_fallback and voice != fallback_voice:
                print(f"WARNING: No audio received for voice '{voice}' while creating {out_path.name}. Retrying once with '{fallback_voice}'.")
                voice = fallback_voice
                tried_no_audio_fallback = True
                continue

            transient = any(token in message for token in ["503", "WSServerHandshakeError", "Invalid response status", "429", "502", "504"])
            if attempt >= TTS_MAX_RETRIES or not transient:
                raise
            wait_seconds = min(TTS_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), TTS_RETRY_MAX_SECONDS)
            print(f"Transient TTS failure for {out_path.name} on attempt {attempt}/{TTS_MAX_RETRIES}: {exc}. Retrying in {wait_seconds:.1f}s...")
            await asyncio.sleep(wait_seconds)
    if last_error is not None:
        raise last_error


def make_silence_mp3(path: Path, duration_seconds: float) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=24000:cl=mono",
            "-t",
            f"{duration_seconds:.3f}",
            "-acodec",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def get_audio_duration_seconds(path: Path) -> float:
    """Return an MP3 duration using ffprobe for dynamic consecutive pauses."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def calculate_consecutive_pause_seconds(source_audio_path: Path) -> float:
    source_duration = get_audio_duration_seconds(source_audio_path)
    dynamic_pause = source_duration * CONSECUTIVE_PAUSE_RATIO
    return max(CONSECUTIVE_MIN_PAUSE, min(CONSECUTIVE_MAX_PAUSE, dynamic_pause))


def unique_nonempty(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = normalize_space(value)
        if not value:
            continue
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(path.read_text(encoding="latin-1"))


def normalize_speaker_key(value: Any) -> str:
    return normalize_space(str(value or "")).casefold()


def get_audio_profile(data: dict[str, Any]) -> dict[str, Any]:
    profile = data.get("audio_profile")
    if not isinstance(profile, dict):
        profile = {}

    # Backwards/transition support: allow these blocks at top level too.
    if "voice_assignments" not in profile and isinstance(data.get("voice_assignments"), dict):
        profile = {**profile, "voice_assignments": data.get("voice_assignments")}
    if "default_voices" not in profile and isinstance(data.get("default_voices"), dict):
        profile = {**profile, "default_voices": data.get("default_voices")}

    return profile



def has_audio_profile_or_voice_assignments(data: dict[str, Any]) -> bool:
    profile = data.get("audio_profile")
    if isinstance(profile, dict) and profile:
        return True
    if isinstance(data.get("voice_assignments"), dict) and data.get("voice_assignments"):
        return True
    if isinstance(data.get("default_voices"), dict) and data.get("default_voices"):
        return True
    return False


def outputs_are_fresh(script_path: Path, outputs: list[Path]) -> bool:
    """True when every expected output exists and is newer than the source JSON.

    This prevents an older MP3 generated from a previous JSON version from being
    reused after you edit the JSON, add audio_profile, or change voices.
    """
    if not outputs or not all(p.exists() for p in outputs):
        return False
    if not REBUILD_IF_JSON_NEWER_THAN_OUTPUT:
        return True
    source_mtime = script_path.stat().st_mtime
    return all(p.stat().st_mtime >= source_mtime for p in outputs)

def normalize_voice_assignments(assignments: Any) -> dict[str, Any]:
    if not isinstance(assignments, dict):
        return {}
    return {normalize_speaker_key(k): v for k, v in assignments.items()}


def _voice_from_entry(entry: Any, lang: str) -> str:
    if isinstance(entry, str):
        return normalize_space(entry)
    if isinstance(entry, dict):
        # Prefer language-specific voice names, but allow a generic fallback.
        return normalize_space(entry.get(lang) or entry.get("voice") or entry.get("default") or "")
    return ""


def get_voice_for_segment(data: dict[str, Any], segment: dict[str, Any], lang: str, fallback_voice: str) -> str:
    """Return a TTS voice for one segment/language chunk.

    Fallback order:
      1. segment-level voice / voices / voice_overrides
      2. speaker-level audio_profile.voice_assignments
      3. audio_profile.default_voices
      4. global EN_VOICE / ES_VOICE constants

    Older JSON files do not need any of these fields.
    """
    if not USE_SPEAKER_VOICE_ASSIGNMENTS:
        return fallback_voice

    lang = str(lang).strip().lower()
    profile = get_audio_profile(data)

    # Segment-level override options.
    segment_voice = segment.get("voice")
    if segment_voice:
        voice = _voice_from_entry(segment_voice, lang)
        if voice:
            return voice

    for key in ("voices", "voice_overrides"):
        segment_voices = segment.get(key)
        if isinstance(segment_voices, dict):
            voice = normalize_space(segment_voices.get(lang) or "")
            if voice:
                return voice

    speaker = segment.get("speaker", "")

    # Optional realism mode: when a chunk is not the original/source language,
    # use the INTERPRETER voice assignment if one exists.
    source_language = str(segment.get("source_language") or segment.get("original_language") or "").strip().lower()
    if INTERPRETED_AUDIO_USES_INTERPRETER_VOICE and source_language and lang != source_language:
        speaker = "INTERPRETER"

    assignments = normalize_voice_assignments(profile.get("voice_assignments"))
    speaker_key = normalize_speaker_key(speaker)
    if speaker_key in assignments:
        voice = _voice_from_entry(assignments[speaker_key], lang)
        if voice:
            return voice

    default_voices = profile.get("default_voices")
    if isinstance(default_voices, dict):
        voice = normalize_space(default_voices.get(lang) or "")
        if voice:
            return voice

    return fallback_voice




def _bool_from_json(value: Any, default: bool) -> bool:
    """Interpret optional JSON booleans without breaking older files."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value_norm = value.strip().casefold()
        if value_norm in {"1", "true", "yes", "y", "on"}:
            return True
        if value_norm in {"0", "false", "no", "n", "off"}:
            return False
    return default


def should_speak_speaker_labels(data: dict[str, Any]) -> bool:
    """Return whether speaker labels should be spoken in script audio.

    Default is controlled by SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO. New JSONs can
    override this per script with audio_profile.speak_speaker_labels.
    """
    profile = get_audio_profile(data)
    if "speak_speaker_labels" in profile:
        return _bool_from_json(profile.get("speak_speaker_labels"), SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO)
    if "speak_speaker_labels" in data:
        return _bool_from_json(data.get("speak_speaker_labels"), SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO)
    return SPEAK_SPEAKER_LABELS_IN_SCRIPT_AUDIO


def strip_leading_speaker_label(text: str, speaker: Any = "") -> str:
    """Remove an initial label like 'JUDGE:' or 'Probation Officer:' from TTS text.

    The JSON should keep the structured speaker field. This helper only cleans
    the spoken text, and only at the very beginning of the chunk.
    """
    text = normalize_space(text)
    if not text:
        return text

    speaker_text = normalize_space(str(speaker or ""))
    candidates = []
    if speaker_text:
        candidates.append(re.escape(speaker_text))
        candidates.append(re.escape(speaker_text.title()))
        candidates.append(re.escape(speaker_text.upper()))

    common = [
        "JUDGE", "THE COURT", "COURT", "PROSECUTOR", "STATE", "DEFENSE ATTORNEY",
        "DEFENSE COUNSEL", "ATTORNEY", "DEFENDANT", "WITNESS", "INTERPRETER",
        "PROBATION OFFICER", "PARENT", "MOTHER", "FATHER", "JUVENILE", "MINOR",
        "GUARDIAN", "CONSERVATOR", "INTERESTED PERSON", "COURT CLERK", "BAILIFF",
        "AGENTE", "MADRE", "PADRE", "MENOR", "INTÉRPRETE", "INTERPRETE",
    ]
    candidates.extend(re.escape(x) for x in common)

    pattern = r"^(?:" + "|".join(dict.fromkeys(candidates)) + r")\s*[:\-–—]\s*"
    cleaned = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)
    return normalize_space(cleaned)


def text_for_tts(data: dict[str, Any], segment: dict[str, Any], lang: str) -> str:
    """Prepare segment text for TTS, optionally including or suppressing labels."""
    text = normalize_space(segment.get(lang, ""))
    speaker = normalize_space(segment.get("speaker", ""))
    if not text:
        return text

    if should_speak_speaker_labels(data):
        # Avoid duplicating an already-written label.
        return text if not speaker or strip_leading_speaker_label(text, speaker) != text else f"{speaker}: {text}"
    return strip_leading_speaker_label(text, speaker)

def print_voice_assignment_summary(data: dict[str, Any], segment_items: list[dict[str, Any]]) -> None:
    if not PRINT_VOICE_ASSIGNMENTS or not USE_SPEAKER_VOICE_ASSIGNMENTS:
        return
    if not has_audio_profile_or_voice_assignments(data):
        return
    speakers = sorted({normalize_space(seg.get("speaker", "")) for seg in segment_items if normalize_space(seg.get("speaker", ""))})
    if not speakers:
        print("Speaker voice mode: audio_profile present, but no speaker labels were found in paired_segments.")
        return
    print("Speaker voice mode enabled. Effective voices:")
    for speaker in speakers:
        probe = {"speaker": speaker, "source_language": "english"}
        en_voice = get_voice_for_segment(data, probe, "english", EN_VOICE)
        es_voice = get_voice_for_segment(data, probe, "spanish", ES_VOICE)
        print(f"  - {speaker}: English={en_voice}; Spanish={es_voice}")

def build_flashcard_lines(terms_used: list[dict[str, Any]]) -> list[tuple[str, list[str]]]:
    """Build TTS-friendly flashcard lines from terms_used.

    The JSON remains the source of truth for display/study data, but these
    returned strings are spoken by TTS. Optional per-term overrides:
      - english_tts: spoken replacement for the English side
      - spanish_tts: spoken replacement for spanish_contextual_choice
    """
    cards: list[tuple[str, list[str]]] = []
    for term in terms_used:
        english_display = normalize_space(term.get("english", ""))
        contextual_display = normalize_space(term.get("spanish_contextual_choice", ""))

        english = normalize_flashcard_tts_text(term.get("english_tts") or english_display, "english")
        contextual = normalize_flashcard_tts_text(term.get("spanish_tts") or contextual_display, "spanish")
        alts_raw = term.get("spanish_alternatives_from_sheet", []) or []

        spanish_options = [contextual]
        if READ_ALL_SPANISH_OPTIONS:
            spanish_options.extend(normalize_flashcard_tts_text(str(x), "spanish") for x in alts_raw)

        spanish_options = unique_nonempty(spanish_options)

        if english and spanish_options:
            cards.append((english, spanish_options))
    return cards


def get_all_spanish_term_candidates(terms_used: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for term in terms_used:
        english = normalize_space(term.get("english", ""))
        contextual = normalize_space(term.get("spanish_contextual_choice", ""))

        if english and contextual:
            key = (english.casefold(), contextual.casefold())
            if key not in seen:
                seen.add(key)
                out.append({"english": english, "spanish": contextual})

        for alt in term.get("spanish_alternatives_from_sheet", []) or []:
            alt = normalize_space(alt)
            if english and alt:
                key = (english.casefold(), alt.casefold())
                if key not in seen:
                    seen.add(key)
                    out.append({"english": english, "spanish": alt})

    return out


def first_appearance_position_in_chunk(item: dict[str, str], english_text: str, spanish_text: str) -> tuple[int, int]:
    en = normalize_space(item.get("english", ""))
    es = normalize_space(item.get("spanish", ""))
    english_cf = english_text.casefold()
    spanish_cf = spanish_text.casefold()

    en_pos = english_cf.find(en.casefold()) if en else -1
    es_pos = spanish_cf.find(es.casefold()) if es else -1

    en_sort = en_pos if en_pos >= 0 else 10**9
    es_sort = es_pos if es_pos >= 0 else 10**9
    return (en_sort, es_sort)


def build_target_term_prompt(matches: list[dict[str, str]], english_text: str = "", spanish_text: str = "") -> str:
    ordered_matches = list(matches)
    if english_text or spanish_text:
        ordered_matches.sort(key=lambda item: first_appearance_position_in_chunk(item, english_text, spanish_text))

    spanish_terms = [m["spanish"] for m in ordered_matches if normalize_space(m.get("spanish", ""))]
    spanish_terms = unique_nonempty(spanish_terms)

    if not spanish_terms:
        return ""
    if len(spanish_terms) == 1:
        return spanish_terms[0]
    return ". ".join(spanish_terms)


def find_target_terms_for_segment(
    english_text: str,
    spanish_text: str,
    terms_used: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates = get_all_spanish_term_candidates(terms_used)

    english_cf = english_text.casefold()
    spanish_cf = spanish_text.casefold()

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in candidates:
        en = item["english"]
        es = item["spanish"]

        if len(es) < TARGETED_MIN_TERM_LENGTH:
            continue

        en_hit = en.casefold() in english_cf
        es_hit = es.casefold() in spanish_cf

        condition = (en_hit or es_hit) if TARGETED_MATCH_TERMS_IN_SEGMENTS else True
        if condition:
            key = (en.casefold(), es.casefold())
            if key not in seen:
                seen.add(key)
                matches.append({"english": en, "spanish": es})

    matches.sort(key=lambda x: len(x["spanish"]), reverse=True)
    return matches


def split_into_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []

    protected = text
    placeholders = {
        "Mr.": "Mr<prd>",
        "Mrs.": "Mrs<prd>",
        "Ms.": "Ms<prd>",
        "Dr.": "Dr<prd>",
        "Hon.": "Hon<prd>",
        "Dept.": "Dept<prd>",
        "No.": "No<prd>",
        "Exh.": "Exh<prd>",
        "Ex.": "Ex<prd>",
        "U.S.": "US<prd>",
        "D.A.": "DA<prd>",
        "P.O.": "PO<prd>",
    }

    for src, repl in placeholders.items():
        protected = protected.replace(src, repl)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÜÑ0-9¿¡\"'])", protected)
    restored = []
    for part in parts:
        for src, repl in placeholders.items():
            part = part.replace(repl, src)
        part = normalize_space(part)
        if part:
            restored.append(part)
    return restored


def find_target_terms_for_sentence(
    english_sentence: str,
    spanish_sentence: str,
    terms_used: list[dict[str, Any]],
) -> list[dict[str, str]]:
    candidates = get_all_spanish_term_candidates(terms_used)
    english_cf = english_sentence.casefold()
    spanish_cf = spanish_sentence.casefold()

    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for item in candidates:
        en = item["english"]
        es = item["spanish"]

        if len(es) < TARGETED_MIN_TERM_LENGTH:
            continue

        en_hit = en.casefold() in english_cf
        es_hit = es.casefold() in spanish_cf

        # Inline cue placement should be anchored to the English sentence itself.
        # Spanish-side matching can drift when sentence splitting is not perfectly aligned,
        # which can cause the cue to sound like it appears before the real sentence.
        condition = en_hit if not INLINE_REQUIRE_MATCH_IN_ENGLISH_OR_SPANISH else (en_hit or es_hit)

        if condition:
            key = (en.casefold(), es.casefold())
            if key not in seen:
                seen.add(key)
                matches.append({"english": en, "spanish": es})

    matches.sort(key=lambda x: len(x["spanish"]), reverse=True)
    return matches


def _normalize_playback_order(order: Any) -> list[str]:
    if isinstance(order, list):
        cleaned = [str(x).strip().lower() for x in order]
        if cleaned == ["english", "spanish"] or cleaned == ["spanish", "english"]:
            return cleaned
    return []


def build_segment_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    paired = data.get("paired_segments", [])

    if PREFER_PAIRED_SEGMENTS and isinstance(paired, list) and paired:
        out: list[dict[str, Any]] = []
        for seg in paired:
            if not isinstance(seg, dict):
                continue

            en = normalize_space(seg.get("english", ""))
            es = normalize_space(seg.get("spanish", ""))
            if not en or not es:
                continue

            playback_order = _normalize_playback_order(seg.get("playback_order"))
            if not playback_order:
                segment_type = str(seg.get("segment_type", "")).strip().lower()
                if segment_type == "answer":
                    playback_order = ["spanish", "english"]
                else:
                    playback_order = ["english", "spanish"]

            source_language = str(
                seg.get("source_language")
                or seg.get("original_language")
                or (playback_order[0] if playback_order else "")
            ).strip().lower()

            out.append(
                {
                    "english": en,
                    "spanish": es,
                    "playback_order": playback_order,
                    "segment_type": str(seg.get("segment_type", "")).strip().lower(),
                    "speaker": str(seg.get("speaker", "")).strip(),
                    "source_language": source_language,
                    "original_language": source_language,
                    "voice": seg.get("voice"),
                    "voices": seg.get("voices"),
                    "voice_overrides": seg.get("voice_overrides"),
                }
            )

        if out:
            return out

    english_script = str(data.get("english_script") or "")
    spanish_script = str(data.get("spanish_script") or "")
    en_parts = [normalize_space(x) for x in re.split(r"\n\s*\n", english_script) if normalize_space(x)]
    es_parts = [normalize_space(x) for x in re.split(r"\n\s*\n", spanish_script) if normalize_space(x)]

    return [
        {
            "english": en,
            "spanish": es,
            "playback_order": ["english", "spanish"],
            "segment_type": "",
            "speaker": "",
            "source_language": "english",
            "original_language": "english",
        }
        for en, es in zip(en_parts, es_parts)
    ]



def get_source_language_for_segment(segment: dict[str, Any]) -> str:
    """Return the language that should be played in source-only practice audio."""
    source_language = str(
        segment.get("source_language")
        or segment.get("original_language")
        or ""
    ).strip().lower()
    if source_language in {"english", "spanish"}:
        return source_language

    playback_order = _normalize_playback_order(segment.get("playback_order"))
    if playback_order:
        return playback_order[0]

    return "english"


def get_source_only_rate_for_language(language: str, en_rate: str, es_rate: str) -> str:
    return es_rate if str(language).lower() == "spanish" else en_rate


def build_source_only_groups(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge adjacent source-only entries when JSON chunking splits one speaker.

    The source-only export is intended to sound like one natural source-language
    practice track. Many JSON files split long turns into pedagogical chunks, but
    those boundaries can create awkward pauses when the same speaker is still
    talking. This groups only adjacent entries with the same speaker, source
    language, TTS voice, and rate. Speaker changes and language/voice changes
    still produce a natural break.
    """
    groups: list[dict[str, Any]] = []
    max_chars = max(200, int(SOURCE_ONLY_MAX_MERGED_CHARS or 2800))

    for entry in entries:
        text = normalize_space(entry.get("text", ""))
        if not text:
            continue

        if not SOURCE_ONLY_MERGE_ADJACENT_SAME_SPEAKER or not groups:
            new_group = dict(entry)
            new_group["text"] = text
            new_group["segments"] = [entry.get("segment_number")]
            new_group["break_kind"] = "speaker_change" if groups else "start"
            groups.append(new_group)
            continue

        prev = groups[-1]
        same_speaker = normalize_speaker_key(prev.get("speaker", "")) == normalize_speaker_key(entry.get("speaker", ""))
        same_language = str(prev.get("language", "")).lower() == str(entry.get("language", "")).lower()
        same_voice = normalize_space(prev.get("voice", "")) == normalize_space(entry.get("voice", ""))
        same_rate = normalize_space(prev.get("rate", "")) == normalize_space(entry.get("rate", ""))
        merged_text = normalize_space(str(prev.get("text", "")) + " " + text)

        if same_speaker and same_language and same_voice and same_rate and len(merged_text) <= max_chars:
            prev["text"] = merged_text
            prev.setdefault("segments", []).append(entry.get("segment_number"))
            continue

        new_group = dict(entry)
        new_group["text"] = text
        new_group["segments"] = [entry.get("segment_number")]
        if same_speaker and same_language and same_voice and same_rate:
            new_group["break_kind"] = "continuation"
        elif same_speaker:
            new_group["break_kind"] = "same_speaker"
        else:
            new_group["break_kind"] = "speaker_change"
        groups.append(new_group)

    return groups


def build_targeted_segment_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    paired = build_segment_items(data)
    terms_used = data.get("terms_used", []) or []

    targeted_items: list[dict[str, Any]] = []
    for seg in paired:
        english = normalize_space(seg.get("english", ""))
        spanish = normalize_space(seg.get("spanish", ""))
        if not english or not spanish:
            continue

        matches = find_target_terms_for_segment(english, spanish, terms_used)
        if not matches:
            continue

        targeted_items.append(
            {
                "english": english,
                "spanish": spanish,
                "target_terms": matches,
                "target_prompt_spanish": build_target_term_prompt(matches, english, spanish),
                "speaker": seg.get("speaker", ""),
                "source_language": seg.get("source_language", ""),
                "original_language": seg.get("original_language", ""),
                "voice": seg.get("voice"),
                "voices": seg.get("voices"),
                "voice_overrides": seg.get("voice_overrides"),
            }
        )
    return targeted_items


def build_inline_cue_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    segment_items = build_segment_items(data)
    terms_used = data.get("terms_used", []) or []
    items: list[dict[str, Any]] = []

    for seg in segment_items:
        english = normalize_space(seg.get("english", ""))
        spanish = normalize_space(seg.get("spanish", ""))
        if not english or not spanish:
            continue

        en_sentences = split_into_sentences(english)
        es_sentences = split_into_sentences(spanish)
        sentence_items: list[dict[str, str]] = []
        max_len = max(len(en_sentences), len(es_sentences))

        for i in range(max_len):
            en_sentence = en_sentences[i] if i < len(en_sentences) else ""
            es_sentence = es_sentences[i] if i < len(es_sentences) else ""
            if not en_sentence:
                continue

            matches = find_target_terms_for_sentence(en_sentence, es_sentence, terms_used)
            cue_text = build_target_term_prompt(matches, en_sentence, es_sentence)
            sentence_items.append(
                {
                    "english_sentence": en_sentence,
                    "spanish_cue": cue_text,
                    "has_target": bool(cue_text),
                }
            )

        if sentence_items:
            items.append({
                "sentences": sentence_items,
                "speaker": seg.get("speaker", ""),
                "source_language": seg.get("source_language", ""),
                "original_language": seg.get("original_language", ""),
                "voice": seg.get("voice"),
                "voices": seg.get("voices"),
                "voice_overrides": seg.get("voice_overrides"),
            })

    return items


def create_concat_file(paths: list[Path], concat_path: Path) -> None:
    lines = []
    for p in paths:
        resolved = p.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{resolved}'")
    concat_path.write_text("\n".join(lines), encoding="utf-8")


def concat_mp3s(parts: list[Path], out_path: Path) -> None:
    concat_txt = out_path.with_suffix(".concat.txt")
    create_concat_file(parts, concat_txt)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_txt),
            "-ar",
            "24000",
            "-ac",
            "1",
            "-c:a",
            "libmp3lame",
            "-b:a",
            MP3_BITRATE,
            str(out_path),
        ],
        check=True,
    )
    try:
        concat_txt.unlink()
    except FileNotFoundError:
        pass


def concat_mp3s_with_gap(parts: list[Path], out_path: Path, gap_seconds: float = COMBINED_SCRIPT_GAP) -> None:
    if not parts:
        raise ValueError("No MP3 parts were provided for combined concatenation.")
    with tempfile.TemporaryDirectory(prefix="combined_audio_gap_") as tmp:
        tmpdir = Path(tmp)
        gap_file = tmpdir / "gap.mp3"
        make_silence_mp3(gap_file, gap_seconds)

        expanded_parts: list[Path] = []
        for i, part in enumerate(parts):
            expanded_parts.append(part)
            if i < len(parts) - 1:
                expanded_parts.append(gap_file)

        concat_mp3s(expanded_parts, out_path)


def script_mp3_sort_key(path: Path) -> tuple[str, int]:
    stem = path.stem.upper()
    match = re.match(r"^([A-Z]+(?:-CON)?)-(\d{3})(?:_.*)?$", stem)
    if not match:
        return (stem, 999999)
    prefix, number = match.groups()
    return (prefix, int(number))


def get_all_finished_script_mp3s(output_dir: Path) -> list[Path]:
    finished: list[Path] = []
    for candidate in output_dir.glob("*.mp3"):
        if candidate.name == "combined_audio_all_scripts.mp3":
            continue
        if re.match(r"^[A-Z]+(?:-CON)?-\d{3}(?:_.*)?\.mp3$", candidate.name, flags=re.IGNORECASE):
            finished.append(candidate)
    return sorted(finished, key=script_mp3_sort_key)


def get_finished_script_rendition_mp3s(output_dir: Path) -> list[Path]:
    finished: list[Path] = []
    for candidate in output_dir.glob("*.mp3"):
        if candidate.name.startswith("combined_audio_"):
            continue
        if re.match(r"^[A-Z]+(?:-CON)?-\d{3}_script\.mp3$", candidate.name, flags=re.IGNORECASE):
            finished.append(candidate)
    return sorted(finished, key=lambda p: script_mp3_sort_key(Path(p.stem[:-7])))


async def build_audio_for_script(script_path: Path) -> list[Path]:
    data = load_json(script_path)
    script_id = data.get("script_id", script_path.stem)

    terms_used = data.get("terms_used", []) or []
    segment_items = build_segment_items(data)
    targeted_items = build_targeted_segment_items(data) if GENERATE_TARGETED_HARD_TERMS else []
    inline_cue_items = build_inline_cue_items(data) if GENERATE_INLINE_TERM_CUES else []

    if not segment_items and not terms_used and not targeted_items and not inline_cue_items:
        print(f"Skipping {script_path.name}: no usable terms or script content found.")
        return []

    combined_output_mp3 = OUTPUT_DIR / f"{script_id}.mp3"
    flashcards_output_mp3 = OUTPUT_DIR / f"{script_id}_flashcards.mp3"
    script_output_mp3 = OUTPUT_DIR / f"{script_id}_script.mp3"
    targeted_output_mp3 = OUTPUT_DIR / f"{script_id}_targeted_terms.mp3"
    inline_cues_output_mp3 = OUTPUT_DIR / f"{script_id}_inline_term_cues.mp3"
    full_speed_output_mp3 = OUTPUT_DIR / f"{script_id}_full_speed_increase.mp3"
    consecutive_output_mp3 = OUTPUT_DIR / f"{script_id}_consecutive_mode.mp3"
    source_only_output_mp3 = OUTPUT_DIR / f"{script_id}_source_only.mp3"

    should_force_rebuild_for_profile = FORCE_REBUILD_WHEN_AUDIO_PROFILE_PRESENT and has_audio_profile_or_voice_assignments(data)

    if SPLIT_FLASHCARDS_AND_SCRIPT:
        existing_targets: list[Path] = []
        if INCLUDE_FLASHCARDS and terms_used:
            existing_targets.append(flashcards_output_mp3)
        if segment_items:
            existing_targets.append(script_output_mp3)
        if GENERATE_TARGETED_HARD_TERMS and targeted_items:
            existing_targets.append(targeted_output_mp3)
        if GENERATE_INLINE_TERM_CUES and inline_cue_items:
            existing_targets.append(inline_cues_output_mp3)
        if GENERATE_FULL_SPEED_INCREASE and segment_items:
            existing_targets.append(full_speed_output_mp3)
        if GENERATE_CONSECUTIVE_MODE and segment_items:
            existing_targets.append(consecutive_output_mp3)
        if GENERATE_SOURCE_ONLY_AUDIO and segment_items:
            existing_targets.append(source_only_output_mp3)
        if SKIP_EXISTING_SCRIPT_MP3S and not should_force_rebuild_for_profile and outputs_are_fresh(script_path, existing_targets):
            print(f"Skipping {script_id}: split audio already exists and is up to date.")
            return existing_targets
    else:
        existing_targets = [combined_output_mp3]
        if GENERATE_CONSECUTIVE_MODE and segment_items:
            existing_targets.append(consecutive_output_mp3)
        if GENERATE_SOURCE_ONLY_AUDIO and segment_items:
            existing_targets.append(source_only_output_mp3)
        if SKIP_EXISTING_SCRIPT_MP3S and not should_force_rebuild_for_profile and outputs_are_fresh(script_path, existing_targets):
            print(f"Skipping {script_id}: existing audio is up to date.")
            return existing_targets

    speed = get_speed_config()
    en_rate = speed["en_rate"]
    es_rate = speed["es_rate"]

    print(f"Processing {script_id}...")
    print_voice_assignment_summary(data, segment_items)

    with tempfile.TemporaryDirectory(prefix=f"{script_id}_audio_") as tmp:
        tmpdir = Path(tmp)

        flashcard_en_es_pause = tmpdir / "pause_flashcard_en_es.mp3"
        flashcard_alt_pause = tmpdir / "pause_flashcard_alt.mp3"
        flashcard_term_gap = tmpdir / "pause_flashcard_term_gap.mp3"
        segment_en_es_pause = tmpdir / "pause_segment_en_es.mp3"
        segment_gap = tmpdir / "pause_segment_gap.mp3"
        section_gap = tmpdir / "pause_section_gap.mp3"
        targeted_interpret_pause = tmpdir / "pause_targeted_interpret.mp3"
        targeted_term_pause = tmpdir / "pause_targeted_term.mp3"
        targeted_retry_pause = tmpdir / "pause_targeted_retry.mp3"
        targeted_final_fast_pause = tmpdir / "pause_targeted_final_fast.mp3"
        targeted_segment_gap = tmpdir / "pause_targeted_segment_gap.mp3"
        inline_sentence_pause = tmpdir / "pause_inline_sentence.mp3"
        inline_cue_pause = tmpdir / "pause_inline_cue.mp3"
        inline_segment_gap = tmpdir / "pause_inline_segment_gap.mp3"
        full_speed_interpret_pause_learning = tmpdir / "pause_full_speed_interpret_learning.mp3"
        full_speed_interpret_pause_normal = tmpdir / "pause_full_speed_interpret_normal.mp3"
        full_speed_final_fast_pause = tmpdir / "pause_full_speed_final_fast.mp3"
        full_speed_short_pause = tmpdir / "pause_full_speed_short.mp3"
        full_speed_segment_gap = tmpdir / "pause_full_speed_segment_gap.mp3"
        consecutive_segment_gap = tmpdir / "pause_consecutive_segment_gap.mp3"
        source_only_segment_gap = tmpdir / "pause_source_only_segment_gap.mp3"
        source_only_speaker_change_gap = tmpdir / "pause_source_only_speaker_change_gap.mp3"
        source_only_merged_continuation_gap = tmpdir / "pause_source_only_merged_continuation_gap.mp3"

        make_silence_mp3(flashcard_en_es_pause, FLASHCARD_EN_ES_PAUSE)
        make_silence_mp3(flashcard_alt_pause, FLASHCARD_ALT_PAUSE)
        make_silence_mp3(flashcard_term_gap, FLASHCARD_TERM_GAP)
        make_silence_mp3(segment_en_es_pause, SEGMENT_EN_ES_PAUSE)
        make_silence_mp3(segment_gap, SEGMENT_GAP)
        make_silence_mp3(section_gap, SECTION_GAP)
        make_silence_mp3(targeted_interpret_pause, TARGETED_INTERPRET_PAUSE)
        make_silence_mp3(targeted_term_pause, TARGETED_TERM_PAUSE)
        make_silence_mp3(targeted_retry_pause, TARGETED_RETRY_PAUSE)
        make_silence_mp3(targeted_final_fast_pause, TARGETED_FINAL_FAST_PAUSE)
        make_silence_mp3(targeted_segment_gap, TARGETED_SEGMENT_GAP)
        make_silence_mp3(inline_sentence_pause, INLINE_SENTENCE_PAUSE)
        make_silence_mp3(inline_cue_pause, INLINE_CUE_PAUSE)
        make_silence_mp3(inline_segment_gap, INLINE_SEGMENT_GAP)
        make_silence_mp3(full_speed_interpret_pause_learning, FULL_SPEED_INTERPRET_PAUSE_LEARNING)
        make_silence_mp3(full_speed_interpret_pause_normal, FULL_SPEED_INTERPRET_PAUSE_NORMAL)
        make_silence_mp3(full_speed_final_fast_pause, FULL_SPEED_FINAL_FAST_PAUSE)
        make_silence_mp3(full_speed_short_pause, FULL_SPEED_SHORT_PAUSE)
        make_silence_mp3(full_speed_segment_gap, FULL_SPEED_SEGMENT_GAP)
        make_silence_mp3(consecutive_segment_gap, CONSECUTIVE_SEGMENT_GAP)
        make_silence_mp3(source_only_segment_gap, SOURCE_ONLY_SEGMENT_GAP)
        make_silence_mp3(source_only_speaker_change_gap, SOURCE_ONLY_SPEAKER_CHANGE_GAP)
        make_silence_mp3(source_only_merged_continuation_gap, SOURCE_ONLY_MERGED_CONTINUATION_GAP)

        counter = 0
        flashcard_parts: list[Path] = []
        script_parts: list[Path] = []
        targeted_parts: list[Path] = []
        inline_cue_parts: list[Path] = []
        full_speed_parts: list[Path] = []
        consecutive_parts: list[Path] = []
        source_only_parts: list[Path] = []
        source_only_entries: list[dict[str, Any]] = []

        if INCLUDE_FLASHCARDS and terms_used:
            cards = build_flashcard_lines(terms_used)
            for english, spanish_options in cards:
                en_file = tmpdir / f"{counter:04d}_card_en.mp3"
                counter += 1
                await tts_to_mp3(english, EN_VOICE, en_rate, en_file)
                flashcard_parts.append(en_file)
                flashcard_parts.append(flashcard_en_es_pause)

                for i, spanish in enumerate(spanish_options):
                    es_file = tmpdir / f"{counter:04d}_card_es_{i}.mp3"
                    counter += 1
                    await tts_to_mp3(spanish, ES_VOICE, es_rate, es_file)
                    flashcard_parts.append(es_file)
                    if i < len(spanish_options) - 1:
                        flashcard_parts.append(flashcard_alt_pause)

                flashcard_parts.append(flashcard_term_gap)

            if flashcard_parts:
                flashcard_parts.append(section_gap)

        for idx, seg in enumerate(segment_items, start=1):
            english = text_for_tts(data, seg, "english")
            spanish = text_for_tts(data, seg, "spanish")
            playback_order = seg.get("playback_order") or ["english", "spanish"]

            en_file = tmpdir / f"{counter:04d}_seg_{idx:03d}_en.mp3"
            counter += 1
            es_file = tmpdir / f"{counter:04d}_seg_{idx:03d}_es.mp3"
            counter += 1

            en_voice = get_voice_for_segment(data, seg, "english", EN_VOICE)
            es_voice = get_voice_for_segment(data, seg, "spanish", ES_VOICE)
            await tts_to_mp3(english, en_voice, en_rate, en_file)
            await tts_to_mp3(spanish, es_voice, es_rate, es_file)

            file_map = {"english": en_file, "spanish": es_file}

            for pos, lang in enumerate(playback_order):
                script_parts.append(file_map[lang])
                if pos < len(playback_order) - 1:
                    script_parts.append(segment_en_es_pause)

            script_parts.append(segment_gap)

            if GENERATE_CONSECUTIVE_MODE:
                consecutive_order = _normalize_playback_order(playback_order) or [playback_order[0], playback_order[1]]
                if len(consecutive_order) >= 2:
                    source_lang = consecutive_order[0]
                    interpreted_lang = consecutive_order[1]
                    source_file = file_map[source_lang]
                    interpreted_file = file_map[interpreted_lang]
                    pause_seconds = calculate_consecutive_pause_seconds(source_file)
                    dynamic_pause = tmpdir / f"{counter:04d}_consecutive_{idx:03d}_dynamic_pause.mp3"
                    counter += 1
                    make_silence_mp3(dynamic_pause, pause_seconds)

                    consecutive_parts.append(source_file)
                    consecutive_parts.append(dynamic_pause)
                    consecutive_parts.append(interpreted_file)
                    consecutive_parts.append(consecutive_segment_gap)

            if GENERATE_SOURCE_ONLY_AUDIO:
                source_lang = get_source_language_for_segment(seg)
                source_text = text_for_tts(data, seg, source_lang)
                source_voice = get_voice_for_segment(data, seg, source_lang, ES_VOICE if source_lang == "spanish" else EN_VOICE)
                source_rate = get_source_only_rate_for_language(source_lang, en_rate, es_rate)
                if source_text:
                    source_only_entries.append({
                        "segment_number": seg.get("segment_number", idx),
                        "speaker": seg.get("speaker", ""),
                        "language": source_lang,
                        "voice": source_voice,
                        "rate": source_rate,
                        "text": source_text,
                    })

        if GENERATE_SOURCE_ONLY_AUDIO and source_only_entries:
            source_only_groups = build_source_only_groups(source_only_entries)
            for group_idx, group in enumerate(source_only_groups, start=1):
                if source_only_parts:
                    break_kind = group.get("break_kind")
                    if break_kind == "speaker_change":
                        source_only_parts.append(source_only_speaker_change_gap)
                    elif break_kind == "continuation":
                        source_only_parts.append(source_only_merged_continuation_gap)
                    else:
                        source_only_parts.append(source_only_segment_gap)

                group_file = tmpdir / f"{counter:04d}_source_only_group_{group_idx:03d}.mp3"
                counter += 1
                await tts_to_mp3(group["text"], group["voice"], group["rate"], group_file)
                source_only_parts.append(group_file)

        if GENERATE_TARGETED_HARD_TERMS:
            for idx, item in enumerate(targeted_items, start=1):
                english = text_for_tts(data, item, "english")
                spanish = text_for_tts(data, item, "spanish")
                target_prompt = item["target_prompt_spanish"]
                if not target_prompt:
                    continue

                en1_file = tmpdir / f"{counter:04d}_target_{idx:03d}_en_learning.mp3"
                counter += 1
                target_file = tmpdir / f"{counter:04d}_target_{idx:03d}_terms_es.mp3"
                counter += 1
                en2_file = tmpdir / f"{counter:04d}_target_{idx:03d}_en_normal.mp3"
                counter += 1
                full_es_file = tmpdir / f"{counter:04d}_target_{idx:03d}_full_es.mp3"
                counter += 1
                en3_file = tmpdir / f"{counter:04d}_target_{idx:03d}_en_fast.mp3"
                counter += 1

                en_voice = get_voice_for_segment(data, item, "english", EN_VOICE)
                es_voice = get_voice_for_segment(data, item, "spanish", ES_VOICE)
                await tts_to_mp3(english, en_voice, TARGETED_RATE_LEARNING, en1_file)
                await tts_to_mp3(target_prompt, es_voice, es_rate, target_file)
                await tts_to_mp3(english, en_voice, TARGETED_RATE_NORMAL, en2_file)
                if TARGETED_INCLUDE_FULL_SPANISH_AT_END:
                    await tts_to_mp3(spanish, es_voice, es_rate, full_es_file)
                if TARGETED_INCLUDE_FAST_FINAL_RECALL:
                    await tts_to_mp3(english, en_voice, TARGETED_RATE_FAST, en3_file)

                targeted_parts.append(en1_file)
                targeted_parts.append(targeted_interpret_pause)
                targeted_parts.append(target_file)
                targeted_parts.append(targeted_term_pause)

                if TARGETED_REPEAT_ENGLISH_CHUNK:
                    targeted_parts.append(en2_file)
                    targeted_parts.append(targeted_retry_pause)

                if TARGETED_INCLUDE_FULL_SPANISH_AT_END:
                    targeted_parts.append(full_es_file)

                if TARGETED_INCLUDE_FAST_FINAL_RECALL:
                    targeted_parts.append(targeted_term_pause)
                    targeted_parts.append(en3_file)
                    targeted_parts.append(targeted_final_fast_pause)

                targeted_parts.append(targeted_segment_gap)

        if GENERATE_INLINE_TERM_CUES:
            for seg_idx, seg in enumerate(inline_cue_items, start=1):
                sentence_items = seg.get("sentences", [])
                for sent_idx, sent in enumerate(sentence_items, start=1):
                    english_sentence = sent.get("english_sentence", "")
                    spanish_cue = sent.get("spanish_cue", "")
                    has_target = bool(sent.get("has_target", False))
                    if not english_sentence:
                        continue

                    en_file = tmpdir / f"{counter:04d}_inline_{seg_idx:03d}_{sent_idx:03d}_en.mp3"
                    counter += 1
                    en_voice = get_voice_for_segment(data, seg, "english", EN_VOICE)
                    es_voice = get_voice_for_segment(data, seg, "spanish", ES_VOICE)
                    await tts_to_mp3(english_sentence, en_voice, en_rate, en_file)
                    inline_cue_parts.append(en_file)

                    if has_target and spanish_cue:
                        cue_file = tmpdir / f"{counter:04d}_inline_{seg_idx:03d}_{sent_idx:03d}_cue_es.mp3"
                        counter += 1
                        await tts_to_mp3(spanish_cue, es_voice, es_rate, cue_file)
                        inline_cue_parts.append(inline_sentence_pause)
                        inline_cue_parts.append(cue_file)
                        inline_cue_parts.append(inline_cue_pause)

                inline_cue_parts.append(inline_segment_gap)

        if GENERATE_FULL_SPEED_INCREASE:
            for idx, seg in enumerate(segment_items, start=1):
                english = text_for_tts(data, seg, "english")
                spanish = text_for_tts(data, seg, "spanish")

                en_learning_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_en_learning.mp3"
                counter += 1
                es_learning_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_es_learning.mp3"
                counter += 1
                en_normal_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_en_normal.mp3"
                counter += 1
                es_normal_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_es_normal.mp3"
                counter += 1
                en_fast_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_en_fast.mp3"
                counter += 1

                en_voice = get_voice_for_segment(data, seg, "english", EN_VOICE)
                es_voice = get_voice_for_segment(data, seg, "spanish", ES_VOICE)
                await tts_to_mp3(english, en_voice, FULL_SPEED_RATE_LEARNING, en_learning_file)
                await tts_to_mp3(spanish, es_voice, FULL_SPEED_RATE_LEARNING, es_learning_file)
                await tts_to_mp3(english, en_voice, FULL_SPEED_RATE_NORMAL, en_normal_file)
                await tts_to_mp3(spanish, es_voice, FULL_SPEED_RATE_NORMAL, es_normal_file)
                await tts_to_mp3(english, en_voice, FULL_SPEED_RATE_FAST, en_fast_file)

                # Respect each segment's playback_order in full-speed mode.
                # Earlier versions always used English -> Spanish here, even when
                # paired_segments specified ["spanish", "english"] for Spanish-source answers.
                # This keeps the full_speed_increase drill aligned with the normal script audio.
                playback_order = _normalize_playback_order(seg.get("playback_order")) or ["english", "spanish"]

                learning_files_by_lang = {
                    "english": en_learning_file,
                    "spanish": es_learning_file,
                }
                normal_files_by_lang = {
                    "english": en_normal_file,
                    "spanish": es_normal_file,
                }

                first_lang, second_lang = playback_order

                full_speed_parts.append(learning_files_by_lang[first_lang])
                full_speed_parts.append(full_speed_interpret_pause_learning)
                full_speed_parts.append(learning_files_by_lang[second_lang])
                full_speed_parts.append(full_speed_short_pause)

                full_speed_parts.append(normal_files_by_lang[first_lang])
                full_speed_parts.append(full_speed_interpret_pause_normal)
                full_speed_parts.append(normal_files_by_lang[second_lang])
                full_speed_parts.append(full_speed_short_pause)

                if first_lang == "english":
                    full_speed_parts.append(en_fast_file)
                    full_speed_parts.append(full_speed_final_fast_pause)
                    if FULL_SPEED_INCLUDE_FAST_SPANISH:
                        es_fast_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_es_fast.mp3"
                        counter += 1
                        await tts_to_mp3(spanish, es_voice, FULL_SPEED_RATE_FAST, es_fast_file)
                        full_speed_parts.append(es_fast_file)
                else:
                    es_fast_file = tmpdir / f"{counter:04d}_fullspeed_{idx:03d}_es_fast.mp3"
                    counter += 1
                    await tts_to_mp3(spanish, es_voice, FULL_SPEED_RATE_FAST, es_fast_file)
                    full_speed_parts.append(es_fast_file)
                    full_speed_parts.append(full_speed_final_fast_pause)
                    if FULL_SPEED_INCLUDE_FAST_SPANISH:
                        # Keep the final recall pair in source-language order.
                        # For Spanish-source segments, English is the interpretation.
                        full_speed_parts.append(en_fast_file)

                full_speed_parts.append(full_speed_segment_gap)
        saved_outputs: list[Path] = []

        if SPLIT_FLASHCARDS_AND_SCRIPT:
            if flashcard_parts:
                concat_mp3s(flashcard_parts, flashcards_output_mp3)
                print(f"Saved flashcards: {flashcards_output_mp3}")
                saved_outputs.append(flashcards_output_mp3)
            if script_parts:
                concat_mp3s(script_parts, script_output_mp3)
                print(f"Saved script rendition: {script_output_mp3}")
                saved_outputs.append(script_output_mp3)
            if targeted_parts:
                concat_mp3s(targeted_parts, targeted_output_mp3)
                print(f"Saved targeted hard-terms drill: {targeted_output_mp3}")
                saved_outputs.append(targeted_output_mp3)
            if inline_cue_parts:
                concat_mp3s(inline_cue_parts, inline_cues_output_mp3)
                print(f"Saved inline term cues drill: {inline_cues_output_mp3}")
                saved_outputs.append(inline_cues_output_mp3)
            if full_speed_parts:
                concat_mp3s(full_speed_parts, full_speed_output_mp3)
                print(f"Saved full speed increase drill: {full_speed_output_mp3}")
                saved_outputs.append(full_speed_output_mp3)
            if consecutive_parts:
                concat_mp3s(consecutive_parts, consecutive_output_mp3)
                print(f"Saved consecutive mode drill: {consecutive_output_mp3}")
                saved_outputs.append(consecutive_output_mp3)
            if source_only_parts:
                concat_mp3s(source_only_parts, source_only_output_mp3)
                print(f"Saved source-only practice audio: {source_only_output_mp3}")
                saved_outputs.append(source_only_output_mp3)
            return saved_outputs

        parts: list[Path] = []
        if flashcard_parts:
            parts.extend(flashcard_parts)
        if script_parts:
            parts.extend(script_parts)
        if targeted_parts:
            parts.extend(targeted_parts)
        if inline_cue_parts:
            parts.extend(inline_cue_parts)
        if full_speed_parts:
            parts.extend(full_speed_parts)

        saved_outputs: list[Path] = []
        if consecutive_parts:
            concat_mp3s(consecutive_parts, consecutive_output_mp3)
            print(f"Saved consecutive mode drill: {consecutive_output_mp3}")
            saved_outputs.append(consecutive_output_mp3)
        if source_only_parts:
            concat_mp3s(source_only_parts, source_only_output_mp3)
            print(f"Saved source-only practice audio: {source_only_output_mp3}")
            saved_outputs.append(source_only_output_mp3)

        if not parts:
            return saved_outputs

        concat_mp3s(parts, combined_output_mp3)
        print(f"Saved: {combined_output_mp3}")
        saved_outputs.append(combined_output_mp3)
        return saved_outputs


async def main() -> None:
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg was not found on PATH. Please install ffmpeg first.")

    await initialize_voice_validation()

    files = sorted(p for p in SCRIPTS_DIR.glob("*.json") if p.name != "index.json")
    if not files:
        print("No JSON script files found in scripts/")
        return

    target_names = get_target_names()
    if target_names is not None:
        files = [p for p in files if p.name in target_names]

    if not files:
        print("No matching target files found.")
        return

    # If two JSON files share the same script_id, they would write to the same
    # MP3 filenames. Prefer the version with audio_profile/voice_assignments;
    # otherwise prefer the newest file. This avoids accidentally generating the
    # old default-voice JSON first and then skipping the newer voice-enabled one.
    by_script_id: dict[str, Path] = {}
    for p in files:
        try:
            data = load_json(p)
            sid = str(data.get("script_id") or p.stem)
            has_profile = has_audio_profile_or_voice_assignments(data)
        except Exception:
            sid = p.stem
            has_profile = False
        current = by_script_id.get(sid)
        if current is None:
            by_script_id[sid] = p
            continue
        try:
            current_data = load_json(current)
            current_has_profile = has_audio_profile_or_voice_assignments(current_data)
        except Exception:
            current_has_profile = False
        choose_new = (has_profile and not current_has_profile) or (has_profile == current_has_profile and p.stat().st_mtime > current.stat().st_mtime)
        if choose_new:
            print(f"Duplicate script_id {sid!r}: using {p.name} instead of {current.name}.")
            by_script_id[sid] = p
        else:
            print(f"Duplicate script_id {sid!r}: using {current.name}; skipping {p.name}.")

    files = sorted(by_script_id.values())

    generated_mp3s: list[Path] = []

    for path in files:
        results = await build_audio_for_script(path)
        if results:
            generated_mp3s.extend(results)

    if GENERATE_COMBINED_AUDIO_ALL_SCRIPTS:
        all_script_mp3s = get_finished_script_rendition_mp3s(OUTPUT_DIR) if SPLIT_FLASHCARDS_AND_SCRIPT else get_all_finished_script_mp3s(OUTPUT_DIR)
        if all_script_mp3s:
            combined_out = OUTPUT_DIR / "combined_audio_all_scripts.mp3"
            concat_mp3s_with_gap(all_script_mp3s, combined_out, gap_seconds=COMBINED_SCRIPT_GAP)
            print(f"Saved combined MP3 from all existing script audio: {combined_out}")
        elif generated_mp3s:
            print("No finished script-level MP3 files were found for combined output.")

    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())

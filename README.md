# Interpreter Audio Generator — Web Version

This is the Streamlit Cloud version of the interpreter training app. It is designed for browser-based practice, temporary exam-mode recordings, transcription, rubric-style feedback, and remedial drill JSON generation.

## Main workflow

1. Upload an interpreter-training JSON script.
2. Generate or play source-only practice audio.
3. Use Exam Mode to record a consecutive or simultaneous attempt.
4. Transcribe the saved exam recording with OpenAI transcription.
5. Review the structured transcript: source text, reference interpretation, and your attempt.
6. Generate a rubric score and remedial practice JSON.
7. Download the summary, transcript files, and remedial JSON.
8. Load the remedial JSON in Drill Studio or run it through the audio generator.

## Current major features

- JSON script upload and validation
- Audio generation with Edge TTS
- Source-only practice audio
- Natural same-speaker source-only grouping for simultaneous passages
- Consecutive and simultaneous practice modes
- Exam Mode with browser microphone recording
- OpenAI transcription, with `whisper-1` as the default model
- Structured transcript review
- Rubric score + performance summary
- AI-generated remedial practice JSON based on:
  - source text
  - reference interpretation
  - your transcript
  - WPM data when available
  - project terminology preferences
- Drill Studio with:
  - Term Flashcards from `terms_used`
  - Segment Drills from `paired_segments`
  - compatibility with older interactive drill JSON files
- Downloadable CSV/TXT/DOCX/JSON/ZIP outputs

## Important cloud note

Streamlit Cloud storage is temporary. Generated audio, recordings, transcripts, drill history, and ZIP files may disappear when the app restarts or the session expires. Download any files you want to keep.

## Files in this deployment

- `audio_generator_streamlit_cloud_V2.py` — main Streamlit Cloud app
- `Current_Audio_Generator_Script_JSON_to_MP3_V7.py` — audio generator module used by the app
- `requirements.txt` — Python dependencies
- `packages.txt` — system packages for Streamlit Cloud

## Streamlit Cloud setup

Set the app entry point to:

```text
interpreter_audio_generator_web_V2_phase3C33_readme_update/audio_generator_streamlit_cloud_V2.py
```

Or, if you place the files at the repository root:

```text
audio_generator_streamlit_cloud_V2.py
```

## Python dependencies

The app expects these packages from `requirements.txt`:

```text
streamlit>=1.35
edge-tts>=6.1
pandas>=2.0
openai>=1.0
python-docx>=1.1
```

## System dependency

The app uses `ffmpeg` / `ffprobe` for audio conversion, concatenation, compression, and WPM timing estimates. Streamlit Cloud should install this from `packages.txt`:

```text
ffmpeg
```

## OpenAI API key

For transcription and rubric/remedial JSON generation, provide an OpenAI API key either in the app field or as a Streamlit secret.

Recommended Streamlit secret:

```toml
OPENAI_API_KEY = "your_key_here"
```

The app checks the key in this order:

1. Manual key entered in the app
2. `st.secrets["OPENAI_API_KEY"]`
3. Environment variable `OPENAI_API_KEY`

## Recommended Exam Mode settings

- Basic transcription model: `whisper-1`
- Language hint: `Auto` unless you know the response language
- Rubric AI model: use the strongest available model you are comfortable paying for
- Download the exam ZIP after each full run

## Remedial JSON format

The remedial JSON is intended to be usable as a normal practice script. It should include:

```json
{
  "script_id": "...",
  "category": "Simultaneous Remedial Drill",
  "title": "...",
  "format": "simultaneous_remedial_drill_with_flashcards",
  "audio_profile": {...},
  "terms_used": [...],
  "performance_targets": [...],
  "english_script": "...",
  "spanish_script": "...",
  "paired_segments": [...]
}
```

`terms_used` feeds Term Flashcards. `paired_segments` feeds Segment Drills.

## Deployment checklist

Before pushing to GitHub / redeploying on Streamlit Cloud:

- Confirm `audio_generator_streamlit_cloud_V2.py` is the updated file.
- Confirm `Current_Audio_Generator_Script_JSON_to_MP3_V7.py` is present.
- Confirm `requirements.txt` includes `streamlit`, `edge-tts`, `pandas`, `openai`, and `python-docx`.
- Confirm `packages.txt` includes `ffmpeg`.
- Add `OPENAI_API_KEY` to Streamlit secrets if you do not want to enter it manually.
- Redeploy the Streamlit app.
- Test one short JSON script, one transcription, one rubric/remedial JSON run, and one Drill Studio load.

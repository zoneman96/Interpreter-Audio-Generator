# Interpreter Audio Generator — Web Build

This is the Streamlit Cloud/Web version of the Interpreter Audio Generator.

## Entry point

Use:

```bash
streamlit run audio_generator_streamlit_cloud_V3.py
```

For Streamlit Community Cloud, set the app entry file to:

```text
audio_generator_streamlit_cloud_V3.py
```

## Current web sync

This build syncs the recent Mac UI and exam workflow updates while keeping the app cloud-safe:

- Guided workflow UI
- Compact layout and keyboard shortcuts
- Context Action Panel / compact work area
- Dynamic sidebar settings by workspace
- Drill Studio updates for `terms_used` and `paired_segments`
- Consecutive Exam Mode auto-record workflow when browser permissions allow it
- OpenAI transcription with `whisper-1` default
- Timestamp-anchored review using OpenAI whisper-1 timestamp JSON
- Rubric score + remedial JSON workflow
- AI exam evaluation tracking
- Clean exam review bundle ZIP

## Mac-only features intentionally excluded

The web build does **not** include local whisper.cpp transcription or local Whisper timestamp support. Those remain Mac-only because they require a local binary, model files, and local filesystem paths.

## Required packages

Install Python requirements:

```bash
pip install -r requirements.txt
```

Streamlit Cloud should also install `ffmpeg` from `packages.txt`.

## OpenAI API key

The app checks for the API key in this order:

1. Manual key entered in the app
2. `st.secrets["OPENAI_API_KEY"]`
3. `OPENAI_API_KEY` environment variable

For Streamlit Cloud, add this to Secrets:

```toml
OPENAI_API_KEY = "sk-..."
```

## Temporary storage note

Generated audio, transcripts, score reports, remedial JSON, logs, and exam bundles are stored in a temporary runtime folder. Download any output you want to keep.

## Recommended workflow

1. Upload a JSON script.
2. Generate/play practice audio.
3. Open Exam Mode.
4. Record a consecutive or simultaneous attempt.
5. Transcribe with OpenAI whisper-1.
6. Review transcript/source/reference.
7. Score the attempt and create remedial JSON.
8. Practice that JSON in Drill Studio.
9. Download the exam bundle and remedial practice files.

## Supported JSON structure

The app is designed around JSON scripts with:

- `script_id`
- `title`
- `format`
- `audio_profile`
- `terms_used`
- `english_script`
- `spanish_script`
- `paired_segments`

Older files may still work, but the best Drill Studio and Exam Mode experience comes from scripts with `terms_used` and `paired_segments`.

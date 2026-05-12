# Interpreter Audio Generator — Web 1.2

Cloud-safe Streamlit version of the interpreter training audio/practice app.

## Streamlit Community Cloud entrypoint

```text
audio_generator_streamlit_cloud.py
```

## What is included in Web 1.2

- Upload-based JSON workflow for Streamlit Cloud
- Temporary runtime folders instead of Mac-specific output folders
- Generated-output ZIP downloads
- Consecutive practice with source/correction audio
- Consecutive playback speed controls: 0.85x through 1.50x
- Consecutive audio-only source mode
- Side-by-side source/reference practice view
- Compact segment metadata rows
- Simultaneous practice with paragraph/chunk and full-passage modes
- Simultaneous playback speed controls: 0.85x through 1.50x
- Lag trainer, shadowing mode, and repeat-loop source audio
- Full-passage voice preservation using `audio_profile` when available
- Term review with spoken prompt/answer audio
- Spaced term review / SRS-style review state for the current app session
- Downloadable practice/session logs
- Sample consecutive and simultaneous JSON files in `sample_json/`

## Cloud storage note

The free Streamlit Cloud version should be treated as session-based. Runtime audio, cache files, and progress logs may reset. Download generated ZIP files and CSV logs if you want to keep them.

## Local test

```bash
python -m pip install -r requirements.txt
streamlit run audio_generator_streamlit_cloud.py
```

## System package

`packages.txt` includes `ffmpeg` for audio processing support on Streamlit Community Cloud.

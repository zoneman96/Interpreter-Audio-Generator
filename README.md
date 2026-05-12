# Interpreter Audio Generator — Streamlit Cloud

A Streamlit app for interpreter-training JSON scripts. It can generate audio outputs, create visual practice packets, and run an in-session consecutive practice tool with source/correction audio.

## Files in this repository

```text
audio_generator_streamlit_cloud.py
Current_Audio_Generator_Script_JSON_to_MP3_V7.py
generate_practice_docx.py
requirements.txt
packages.txt
README.md
.gitignore
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run audio_generator_streamlit_cloud.py
```

You also need `ffmpeg` installed locally. On macOS with Homebrew:

```bash
brew install ffmpeg
```

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository.
2. Upload the files listed above.
3. In Streamlit Community Cloud, create a new app from the GitHub repo.
4. Set the app entrypoint to:

```text
audio_generator_streamlit_cloud.py
```

5. Deploy.

`requirements.txt` installs Python dependencies. `packages.txt` installs the Linux system package `ffmpeg` for audio concatenation.

## How the cloud version works

The cloud version does not use a local Mac output folder. Instead:

- Upload JSON files in the sidebar.
- Generate audio and/or practice packets.
- Download the generated ZIP files.
- Use the interactive consecutive practice tab during the current session.
- Download the practice session CSV before closing the browser tab.

Cloud storage is temporary, so do not rely on generated files staying on the server.

## GitHub hygiene

Do not commit generated outputs, virtual environments, cache folders, or the local `.command` launcher. The `.gitignore` file in this package excludes those items.

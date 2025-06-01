# accent_detector
Classifies the english accent

This is a Streamlit web application that analyzes the English accent of a speaker from a **public video URL (.mp4)**. The app downloads the video, extracts audio, transcribes the speech using OpenAI's Whisper model, and classifies the accent into British English, American English, Indian English, or Neutral/Unclassified English based on simple keyword matching.

---

## Features

- Input a public video URL in `.mp4` format
- Download the video and extract audio
- Use Whisper's `small` model to transcribe the audio to text
- Detect the speaker's English accent using keyword rules
- Display detected accent with a confidence score

---

## Installation

Make sure you have Python 3.7+ installed.

Install required Python packages:

```bash
pip install streamlit torch openai-whisper pydub langdetect

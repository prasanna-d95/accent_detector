import streamlit as st  
import os
import tempfile
import torch
import whisper
from pydub import AudioSegment
import urllib.request
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0

# Load Whisper model
model = whisper.load_model("small")

st.set_page_config(page_title="Accent Detector", layout="centered")
st.title(" English Accent Classifier")
st.markdown("Enter a public **video URL (.mp4)** to detect the English accent of the speaker.")

# Input field
video_url = st.text_input("Enter Public Video URL (MP4):", "")

if st.button("Analyze Accent"):
    if not video_url:
        st.error(" Please enter a valid video URL.")
    else:
        tmp_video_path = None
        tmp_audio_path = None

        # Download video
        with st.spinner(" Downloading video..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                    urllib.request.urlretrieve(video_url, tmp_video.name)
                    tmp_video_path = tmp_video.name
                st.success("Video downloaded successfully!")
            except Exception as e:
                st.error(f" Error downloading video: {e}")
                st.stop()

        # Extract audio
        with st.spinner("Extracting audio..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
                    video = AudioSegment.from_file(tmp_video_path)
                    video = video.set_channels(1).set_frame_rate(16000)
                    video.export(tmp_audio.name, format="wav")
                    tmp_audio_path = tmp_audio.name
                st.success(" Audio extracted.")
            except Exception as e:
                st.error(f" Error extracting audio: {e}")
                st.stop()

        # Transcribe and detect accent
        with st.spinner("Analyzing speech for accent..."):
            try:
                result = model.transcribe(tmp_audio_path, language="en")
                text = result.get("text", "").lower()

                # Very basic accent rule mapping (for demo only — real systems use ML-based detection)
                american_keywords = ["gonna", "wanna", "gotta", "ain't"]
                british_keywords = ["flat", "lorry", "boot", "petrol", "mate", "queue"]
                indian_keywords = ["kindly", "revert", "prepone", "passed out", "only", "doubt"]

                detected_accent = "Unknown English Accent"
                if any(word in text for word in british_keywords):
                    detected_accent = "British English"
                elif any(word in text for word in indian_keywords):
                    detected_accent = "Indian English"
                elif any(word in text for word in american_keywords):
                    detected_accent = "American English"
                else:
                    detected_accent = "Neutral or Unclassified English"

                confidence_score = min(100, int(result["segments"][0]["no_speech_prob"] * 100))
                confidence_score = 100 - confidence_score  # Higher = more confident

                st.markdown("### Detected Accent")
                st.success(f"**Detected Accent**: {detected_accent}")
                st.info(f"**Confidence Score**: {confidence_score:.2f}%")

            except Exception as e:
                st.error(f" Error during accent analysis: {e}")

        # Cleanup
        try:
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.unlink(tmp_video_path)
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.unlink(tmp_audio_path)
        except Exception as e:
            st.warning(f"Cleanup warning: {e}")
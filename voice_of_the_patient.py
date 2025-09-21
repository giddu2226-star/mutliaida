import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import os
import subprocess
import tempfile
import time

import assemblyai as aai
from assemblyai import Transcriber, TranscriptionConfig
from langdetect import detect  # ✅ new: backup language detector

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -------------------------------
# Record Audio (optional local mic)
# -------------------------------
def record_audio(file_path, timeout=20, phrase_time_limit=None):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")

            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")

            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k")
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred while recording: {e}")


# -------------------------------
# Transcription with AssemblyAI (Multilingual with fallback detection)
# -------------------------------
def transcribe_with_assemblyai(audio_filepath: str):
    """
    Transcribe audio using AssemblyAI with auto language detection.
    Adds backup detection with langdetect if AssemblyAI mislabels as 'en'.
    """
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise ValueError("ASSEMBLYAI_API_KEY is not set")

    aai.settings.api_key = api_key
    transcriber = Transcriber()
    config = TranscriptionConfig(language_detection=True)

    # 🔹 Convert to PCM WAV (mono, 16kHz)
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_filepath, "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", tmp_wav.name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        logging.info(f"Converted {audio_filepath} → {tmp_wav.name} for AssemblyAI")
    except Exception as e:
        logging.error(f"ffmpeg conversion failed: {e}")
        raise

    # 🔹 Submit async job + wait
    transcript = transcriber.submit(tmp_wav.name, config=config)
    logging.info(f"AssemblyAI job submitted: {transcript.id}")

    transcript = transcript.wait_for_completion()

    if transcript.status == "completed":
        text = getattr(transcript, "text", "")
        detected_lang = getattr(transcript, "language_code", "en")

        # ✅ Backup detection if AssemblyAI says "en" but text is clearly another language
        try:
            if detected_lang == "en" and text:
                backup_lang = detect(text)
                logging.info(f"Langdetect override: {detected_lang} → {backup_lang}")
                detected_lang = backup_lang
        except Exception as e:
            logging.error(f"Langdetect failed, keeping AssemblyAI lang: {e}")

        logging.info(f"Final detected language: {detected_lang}")
        return text, detected_lang
    else:
        raise RuntimeError(f"AssemblyAI transcription failed: {transcript.error}")
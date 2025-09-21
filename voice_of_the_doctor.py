import os
import logging
from gtts import gTTS
from elevenlabs import save
from elevenlabs.client import ElevenLabs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID")  # your cloned voice id if any


# -------------------------------
# Google TTS (fallback)
# -------------------------------
def text_to_speech_with_gtts(input_text: str, output_filepath: str = "final.mp3", lang: str = "en"):
    try:
        tts = gTTS(text=input_text, lang=lang, slow=False)
        tts.save(output_filepath)
        logging.info(f"gTTS audio saved at {output_filepath} (lang={lang})")
        return output_filepath
    except Exception as e:
        logging.error(f"gTTS failed: {e}")
        raise


# -------------------------------
# ElevenLabs TTS (primary)
# -------------------------------
def text_to_speech_with_elevenlabs(input_text: str, output_filepath: str = "final.mp3", lang: str = "en"):
    if not ELEVENLABS_API_KEY:
        logging.warning("ELEVENLABS_API_KEY not found. Falling back to gTTS.")
        return text_to_speech_with_gtts(input_text, output_filepath, lang)

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

        model = "eleven_multilingual_v2" if lang != "en" else "eleven_turbo_v2"
        voice = ELEVENLABS_VOICE_ID if ELEVENLABS_VOICE_ID else "Aria"

        audio = client.generate(
            text=input_text,
            voice=voice,
            model=model,
            output_format="mp3_22050_32"
        )

        save(audio, output_filepath)
        logging.info(f"ElevenLabs audio saved at {output_filepath} (voice={voice}, model={model})")
        return output_filepath

    except Exception as e:
        logging.error(f"ElevenLabs TTS failed: {e}. Falling back to gTTS.")
        return text_to_speech_with_gtts(input_text, output_filepath, lang)
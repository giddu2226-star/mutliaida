from dotenv import load_dotenv
load_dotenv()

import os
import logging
import gradio as gr
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from deep_translator import GoogleTranslator  # ✅ new translator

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_assemblyai
from voice_of_the_doctor import text_to_speech_with_elevenlabs

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Doctor system prompt (base in English)
base_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def process_inputs(audio_input, image_filepath, progress=gr.Progress()):
    try:
        speech_to_text_output = ""
        detected_lang = "en"   # default fallback
        patient_voice_mp3 = None
        doctor_voice_mp3 = None
        doctor_response = ""

        # -------------------------------
        # Step 1: Handle Voice (Mic Input)
        # -------------------------------
        if audio_input is not None:
            logging.info("🎤 Patient voice received from microphone")

            # ✅ Handle tuple (sample_rate, numpy_array) or plain array
            if isinstance(audio_input, tuple):
                sample_rate = audio_input[0]   # first element is sample_rate
                audio_data = audio_input[1]    # second element is numpy array
                logging.info(f"✅ Tuple format detected. sample_rate={sample_rate}, shape={getattr(audio_data, 'shape', None)}")
            else:
                audio_data = audio_input
                sample_rate = 44100  # fallback default
                logging.info("✅ Array format detected. sample_rate=44100")

            # ✅ Ensure audio is always 2D (samples, channels)
            if hasattr(audio_data, "ndim") and audio_data.ndim == 1:
                audio_data = np.expand_dims(audio_data, axis=1)
                logging.info("✅ Expanded mono to 2D array")

            # Save to temporary WAV file
            wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
            sf.write(wav_path, audio_data, sample_rate)
            logging.info(f"Temporary WAV saved: {wav_path}")

            # Convert to MP3 for playback
            patient_voice_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            subprocess.run(
                ["ffmpeg", "-y", "-i", wav_path, "-ar", "44100", "-ac", "2", patient_voice_mp3],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logging.info(f"Patient voice converted to MP3: {patient_voice_mp3}")

            # Transcribe
            progress(0.2, desc="⏳ Transcribing speech with AssemblyAI...")
            speech_to_text_output, detected_lang = transcribe_with_assemblyai(wav_path)

        else:
            logging.info("⚠️ No patient voice provided.")

        # -------------------------------
        # Step 2: Translate base prompt
        # -------------------------------
        progress(0.4, desc=f"🌍 Preparing doctor instructions in {detected_lang}...")
        try:
            translated_prompt = GoogleTranslator(source="auto", target=detected_lang).translate(base_prompt)
        except Exception as e:
            logging.error(f"Translation failed, using English prompt: {e}")
            translated_prompt = base_prompt

        # -------------------------------
        # Step 3: Doctor reasoning
        # -------------------------------
        progress(0.6, desc="🤔 Doctor is analyzing...")
        if image_filepath:
            query_text = f"{translated_prompt}\n\n"
            if speech_to_text_output:
                query_text += f"The patient spoke in {detected_lang}. Patient said: {speech_to_text_output}"
            else:
                query_text += "No speech provided, please analyze only the image."

            doctor_response = analyze_image_with_query(
                query=query_text,
                encoded_image=encode_image(image_filepath),
                model="meta-llama/llama-4-scout-17b-16e-instruct"
            )
        elif speech_to_text_output:
            doctor_response = f"I heard you say: {speech_to_text_output}. But no image was provided for analysis."
        else:
            doctor_response = "⚠️ No input provided (neither speech nor image)."

        # -------------------------------
        # Step 4: TTS (Doctor Reply)
        # -------------------------------
        if doctor_response:
            progress(0.8, desc="🗣 Converting doctor’s reply to speech...")
            doctor_voice_mp3 = text_to_speech_with_elevenlabs(
                input_text=doctor_response,
                output_filepath="doctor_final.mp3",
                lang=detected_lang
            )

        progress(1.0, desc="✅ Done")
        return (
            speech_to_text_output or "No speech provided.",
            detected_lang,
            doctor_response,
            patient_voice_mp3,
            doctor_voice_mp3,
        )

    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        return f"Pipeline error: {e}", "error", "error", None, None


# -------------------------------
# Gradio Interface
# -------------------------------
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(sources=["microphone"], type="numpy", label="🎤 Patient Voice (Speak here)"),
        gr.Image(type="filepath", label="🩺 Upload Medical Image")
    ],
    outputs=[
        gr.Textbox(label="🗣 Speech to Text"),
        gr.Textbox(label="🌍 Detected Language"),
        gr.Textbox(label="👨‍⚕️ Doctor's Response"),
        gr.Audio(type="filepath", label="🔊 Patient Voice (playback)"),
        gr.Audio(type="filepath", label="🔊 Doctor's Reply")
    ],
    title="AI Doctor (Multilingual with AssemblyAI)",
    description="1. Speak about your symptoms in any language.\n2. Upload a medical image.\n3. Click Submit.\nDoctor will reply in your own language."
)

iface.queue().launch(debug=True)
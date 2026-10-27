
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not api_key or api_key == "your_groq_api_key_here":
    raise RuntimeError(
        "GROQ_API_KEY missing hai. .env file mein apni Groq API key set karo."
    )

client = Groq(api_key=api_key)

SYSTEM_PROMPT = """
You are a helpful voice assistant.
Answer the user's actual question clearly and accurately.
Use simple English unless the user speaks Hindi or Hinglish; then respond
naturally in the same language.
Usually answer in 3 to 5 concise sentences.
Give enough explanation to be useful, but avoid unnecessary long answers.
Use complete sentences. Never cut off a sentence intentionally.
If a question needs a step-by-step answer, provide clear numbered steps.
Never claim that you performed an action you did not perform.
"""


def get_answer(user_text: str) -> str:
    user_text = user_text.strip()

    if not user_text:
        return "Please enter a message or speak into the microphone."

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        max_completion_tokens=300,
        temperature=0.4,
    )

    answer = response.choices[0].message.content

    if not answer or not answer.strip():
        return "Sorry, I could not generate an answer. Please try again."

    return answer.strip()


def transcribe_audio(audio_file) -> str:
    """
    audio_file can be a file-like object containing an audio recording.
    Groq Whisper converts the recording to text.
    """
    audio_file.seek(0)

    transcription = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        response_format="text",
    )

    if isinstance(transcription, str):
        return transcription.strip()

    return getattr(transcription, "text", "").strip()

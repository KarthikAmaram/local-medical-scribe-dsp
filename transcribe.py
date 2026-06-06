import whisper
from record_audio import record_audio
from nlp_parser import generate_hpi

print("Loading Whisper model...")
model = whisper.load_model("base")

def run_full_pipeline(duration=70):
    audio_stream = record_audio(duration=duration)
    print("Processing audio stream through Whisper...")
    result = model.transcribe(audio_stream)
    transcribed_text = result["text"]
    final_note = generate_hpi(transcribed_text)
    return transcribed_text, final_note
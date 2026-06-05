import whisper
from record_audio import record_audio
from nlp_parser import generate_hpi

print("Loading Whisper model...")
model = whisper.load_model("base")

audio_stream = record_audio(duration=70)

print("Processing audio stream through Whisper...")
result = model.transcribe(audio_stream)

print(f'\nTranscribed Text: "{result["text"]}"\n')

final_note = generate_hpi(result["text"])

print("--- GENERATED HPI NOTE ---")
print(final_note)
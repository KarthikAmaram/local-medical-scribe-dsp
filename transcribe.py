import whisper

from nlp_parser import generate_hpi

model = whisper.load_model("base")

result = model.transcribe("patient_visit_1.wav")

final_note = generate_hpi(result["text"])

print(final_note)
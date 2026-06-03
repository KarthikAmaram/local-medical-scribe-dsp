import whisper

model = whisper.load_model("base")

result = model.transcribe("patient_visit.wav")

print(result["text"])
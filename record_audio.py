import sounddevice as sd
import numpy as npwhere
from scipy.io import wavfile

#CONFIGURATION
sample_rate = 44100  # Sample rate in Hz
seconds = 5  # Duration of recording in seconds
filename = "patient_visit.wav"  # Output filename

#Record audio
print("Recording started... Speak into the microphone!")

audio_data = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')

sd.wait()

print("Recording finished. Saving audio...")

wavfile.write(filename, sample_rate, audio_data)

print(f"Success! Audio saved as {filename}")
import sounddevice as sd

def record_audio(duration=70):
    sample_rate = 16000  
    
    print("Recording started... Speak into the microphone!")
    
    audio_data = sd.rec(
        int(duration * sample_rate), 
        samplerate=sample_rate, 
        channels=1, 
        dtype='float32'
    )
    sd.wait()
    
    print("Recording finished.")
    
    return audio_data.flatten()
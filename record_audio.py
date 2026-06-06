import sounddevice as sd
import time

recording_active = False

def record_audio(duration=70):
    global recording_active
    recording_active = True
    
    sample_rate = 16000  
    print("Recording started... Speak into the microphone!")
    
    total_samples = int(duration * sample_rate)
    audio_data = sd.rec(
        total_samples, 
        samplerate=sample_rate, 
        channels=1, 
        dtype='float32'
    )
    
    start_time = time.time()
    while recording_active and (time.time() - start_time) < duration:
        time.sleep(0.1)
        
    elapsed_time = time.time() - start_time
    recorded_samples = min(int(elapsed_time * sample_rate), total_samples)
    
    print("Recording finished.")
    sd.stop()
    
    return audio_data[:recorded_samples].flatten()
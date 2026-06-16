import time
import numpy as np
from record_audio import start_recording_stream, stop_recording_stream

def test_system():
    print("--- Starting 3-Second Test Loop ---")
    stream, ring_buffer = start_recording_stream(duration=3)
    
    time.sleep(3)
    
    stop_recording_stream(stream)
    
    available_samples = ring_buffer.size
    print(f"Buffer populated with {available_samples} samples.")
    
    raw_audio = ring_buffer.read(available_samples)
    
    if raw_audio is not None and len(raw_audio) > 0:
        print(f"SUCCESS: Extracted {len(raw_audio)} samples from Ring Buffer.")
        print(f"Audio array bounds: Min = {np.min(raw_audio):.4f}, Max = {np.max(raw_audio):.4f}")
        if np.abs(np.max(raw_audio)) > 0.0001:
            print("SUCCESS: Microphone signal data is non-silent.")
        else:
            print("WARNING: Audio array is silent. Check microphone permissions or hardware mute.")
    else:
        print("FAILURE: No data retrieved from Ring Buffer.")

if __name__ == "__main__":
    test_system()
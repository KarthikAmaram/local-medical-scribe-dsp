import sounddevice as sd
import numpy as np
from dsp_engine import AudioRingBuffer

DEBUG = False

recording_active = False
ring_buffer = None
sample_rate = 16000
chunk_size = 512

def _get_shared_device_index():
    return sd.default.device[0]

def _audio_callback(indata, frames, time, status):
    if recording_active and ring_buffer is not None:
        try:
            ring_buffer.write(indata[:, 0].astype(np.float32))
        except OverflowError:
            pass

def start_recording_stream(duration=70):
    global recording_active, ring_buffer
    
    device_index = _get_shared_device_index()
    device_info = sd.query_devices(device_index, 'input')
    max_channels = int(device_info['max_input_channels'])
    
    buffer_capacity = sample_rate * (duration + 5)
    ring_buffer = AudioRingBuffer(capacity=buffer_capacity)
    recording_active = True
    
    stream = sd.InputStream(
        device=device_index,
        samplerate=sample_rate,
        channels=max_channels,
        blocksize=chunk_size,
        dtype='float32',
        callback=_audio_callback
    )
    
    stream.start()
    if DEBUG:
        print(f"Real-time shared stream active on device index {device_index}... Speak now!")
    return stream, ring_buffer

def stop_recording_stream(stream):
    global recording_active
    recording_active = False
    stream.stop()
    stream.close()
    if DEBUG:
        print("Recording stream stopped cleanly.")
import time
from faster_whisper import WhisperModel
from record_audio import start_recording_stream, stop_recording_stream
from dsp_engine import OverlapAddProcessor
from nlp_parser import generate_hpi

print("Loading high-accuracy Whisper model...")
model = WhisperModel("base", device="cpu", compute_type="float32", cpu_threads=4)
ola_processor = OverlapAddProcessor(frame_size=1024, overlap=512)

is_recording = False

def stop_pipeline_early():
    global is_recording
    is_recording = False

def run_full_pipeline(duration=70):
    global is_recording
    
    stream, ring_buffer = start_recording_stream(duration=duration)
    is_recording = True
    
    start_time = time.time()
    while is_recording and (time.time() - start_time) < duration:
        time.sleep(0.05)
        
    stop_recording_stream(stream)
    is_recording = False
    
    available_samples = ring_buffer.size
    total_samples = (available_samples // 512) * 512
    
    raw_audio = ring_buffer.read(total_samples)
    if raw_audio is None or len(raw_audio) == 0:
        return "No audio captured.", "Error generating HPI."
        
    print("Processing audio through Windowed Overlap-Add framing...")
    clean_audio = ola_processor.process_stream_chunk(raw_audio, filter_func=None)
    
    print("Processing cleaned stream through Faster-Whisper...")
    segments, info = model.transcribe(
        clean_audio, 
        beam_size=3, 
        condition_on_previous_text=False,
        temperature=0.0
    )
    transcribed_text = " ".join([segment.text for segment in segments])
    
    final_note = generate_hpi(transcribed_text)
    return transcribed_text, final_note
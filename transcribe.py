import time
from faster_whisper import WhisperModel
from record_audio import start_recording_stream, stop_recording_stream
from dsp_engine import OverlapAddProcessor

print("Loading high-accuracy Whisper model...")
model = WhisperModel("small.en", device="cpu", compute_type="int8", cpu_threads=4)
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
        return "No audio captured."

    print("Processing audio through Windowed Overlap-Add framing...")
    t0 = time.time()
    clean_audio = ola_processor.process_stream_chunk(raw_audio, filter_func=None)
    print(f"[TIMING] OLA processing: {time.time() - t0:.2f}s")

    print("Processing cleaned stream through Faster-Whisper...")
    t0 = time.time()
    segments, info = model.transcribe(
        clean_audio,
        beam_size=1,
        condition_on_previous_text=False,
        temperature=0.0,
        initial_prompt="Patient is here for a follow-up appointment. She has been taking metformin and lisinopril daily. Blood pressure today is well controlled. Fasting glucose this morning was within normal range. Labs today include fasting glucose, HbA1c, BMP, and lipid panel. Patient denies any chest pain, shortness of breath, nausea, or dizziness. Up to date with all vaccines and screenings. We will continue current medications and recheck labs in three months."
    )
    transcribed_text = " ".join([segment.text for segment in segments])
    print(f"[TIMING] Whisper transcribe: {time.time() - t0:.2f}s")
    print(f"[RAW WHISPER] {transcribed_text}")

    return transcribed_text
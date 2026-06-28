import threading
import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

SPEECH_LOW_HZ = 300
SPEECH_HIGH_HZ = 3400
SAMPLE_RATE = 16000
FILTER_ORDER = 4


class ButterworthBandpassFilter:
    def __init__(self, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ,
                 sample_rate=SAMPLE_RATE, order=FILTER_ORDER):
        self.sample_rate = sample_rate
        nyquist = sample_rate / 2.0
        low = low_hz / nyquist
        high = high_hz / nyquist
        self.sos = butter(order, [low, high], btype='bandpass', output='sos').astype(np.float64)
        self.zi = sosfilt_zi(self.sos).astype(np.float64)

    def __call__(self, frame):
        frame_f64 = frame.astype(np.float64)
        filtered, self.zi = sosfilt(self.sos, frame_f64, zi=self.zi)
        return filtered.astype(np.float32)


class SpectralAnalyzer:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

    def analyze(self, frame):
        n = len(frame)
        spectrum = np.fft.rfft(frame)
        magnitudes = np.abs(spectrum)
        with np.errstate(divide='ignore'):
            db = 20.0 * np.log10(np.maximum(magnitudes, 1e-10))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
        return freqs, db

    def speech_band_db(self, frame, low_hz=SPEECH_LOW_HZ, high_hz=SPEECH_HIGH_HZ):
        freqs, db = self.analyze(frame)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if not np.any(mask):
            return -np.inf
        return float(np.mean(db[mask]))

class AudioRingBuffer:
    def __init__(self, capacity, dtype=np.float32):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.Lock()

    def write(self, data):
        with self.lock:
            num_samples = len(data)
            if num_samples > self.capacity - self.size:
                raise OverflowError("AudioRingBuffer overflow! DSP thread lagging behind ingestion.")
            
            end_space = self.capacity - self.head
            if num_samples <= end_space:
                self.buffer[self.head:self.head + num_samples] = data
            else:
                self.buffer[self.head:self.capacity] = data[:end_space]
                self.buffer[0:num_samples - end_space] = data[end_space:]
                
            self.head = (self.head + num_samples) % self.capacity
            self.size += num_samples

    def read(self, num_samples):
        with self.lock:
            if self.size < num_samples:
                return None
            
            data = np.empty(num_samples, dtype=self.buffer.dtype)
            end_space = self.capacity - self.tail
            
            if num_samples <= end_space:
                data[:] = self.buffer[self.tail:self.tail + num_samples]
            else:
                data[:end_space] = self.buffer[self.tail:self.capacity]
                data[end_space:] = self.buffer[0:num_samples - end_space]
                
            self.tail = (self.tail + num_samples) % self.capacity
            self.size -= num_samples
            return data

class OverlapAddProcessor:
    def __init__(self, frame_size=1024, overlap=512):
        self.frame_size = frame_size
        self.overlap = overlap
        self.hop_size = frame_size - overlap
        self.window = np.hanning(frame_size).astype(np.float32)
        self.ola_buffer = np.zeros(self.overlap, dtype=np.float32)

    def process_stream_chunk(self, linear_audio, filter_func=None):
        num_samples = len(linear_audio)
        assert num_samples % self.hop_size == 0
        
        num_frames = num_samples // self.hop_size
        output_audio = np.zeros(num_samples, dtype=np.float32)
        workspace = np.concatenate([self.ola_buffer, linear_audio])
        
        for f in range(num_frames):
            start_idx = f * self.hop_size
            end_idx = start_idx + self.frame_size
            frame = workspace[start_idx:end_idx]
            windowed_frame = frame * self.window
            
            if filter_func:
                processed_frame = filter_func(windowed_frame)
            else:
                processed_frame = windowed_frame
                
            out_start = f * self.hop_size
            out_end = out_start + self.hop_size
            output_audio[out_start:out_end] = processed_frame[:self.hop_size]
            
            if f < num_frames - 1:
                workspace[out_end:out_end + self.overlap] += processed_frame[self.hop_size:]
                
        self.ola_buffer[:] = workspace[-self.overlap:]
        return output_audio
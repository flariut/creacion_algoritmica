#!/usr/bin/env python3
"""
Fixed 4-track probabilistic mono sample sequencer.

Notes:
- No panning, no delay.
- Each Step has 'on' flag + probability.
- 4-pole lowpass per triggered sample.
- Fixed bug that caused ValueError when removing PlayEvent objects.
- Use: pip install numpy soundfile sounddevice
- Put samples in ./sounds and run: python3 seq2.py
"""
import os
import sys
import math
import random
import asyncio
import threading
from dataclasses import dataclass, field
from typing import List
import numpy as np
import soundfile as sf
import sounddevice as sd

# ----------------------------- Configuration -----------------------------
SOUNDS_DIR = './sounds'
BPM = 132
STEPS = 16
TRACKS = 2
REPEAT_MIN, REPEAT_MAX = 4, 16
SR = 44100
BLOCKSIZE = 2048
LATENCY = 'low'
TRACK_TO_CHANNEL = list(range(TRACKS))

# ----------------------------- DSP Helpers ------------------------------

def semitone_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)

def resample_linear(signal: np.ndarray, ratio: float) -> np.ndarray:
    if ratio == 1.0:
        return signal
    n = signal.shape[0]
    new_n = int(max(1, math.floor(n / ratio)))
    old_idx = np.arange(new_n) * ratio
    i0 = np.floor(old_idx).astype(int)
    frac = old_idx - i0
    i1 = np.minimum(i0 + 1, n - 1)
    if signal.ndim == 1:
        return (1 - frac) * signal[i0] + frac * signal[i1]
    else:
        out = np.zeros((new_n, signal.shape[1]), dtype=signal.dtype)
        for c in range(signal.shape[1]):
            out[:, c] = (1 - frac) * signal[i0, c] + frac * signal[i1, c]
        return out

def apply_gain(signal: np.ndarray, gain: float) -> np.ndarray:
    return signal * gain

def lowpass_4pole(signal: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    # Convert to mono if stereo-ish
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    # 4 cascaded one-pole filters (simple emulation)
    dt = 1.0 / sr
    rc = 1.0 / (2 * math.pi * max(1.0, cutoff))
    alpha = dt / (rc + dt)
    y1 = y2 = y3 = y4 = 0.0
    out = np.zeros_like(signal)
    for i, x in enumerate(signal):
        y1 += alpha * (x - y1)
        y2 += alpha * (y1 - y2)
        y3 += alpha * (y2 - y3)
        y4 += alpha * (y3 - y4)
        out[i] = y4
    return out.astype(np.float32)

# ----------------------------- Sample Loader -----------------------------

@dataclass
class Sample:
    data: np.ndarray
    sr: int
    name: str

def load_samples(folder: str) -> List[Sample]:
    files = []
    if not os.path.isdir(folder):
        print(f"Sounds folder '{folder}' not found.", file=sys.stderr)
        return files
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isfile(path) and fn.lower().endswith(('.wav', '.flac', '.aiff', '.aif')):
            try:
                data, sr = sf.read(path, always_2d=False)
                data = np.asarray(data, dtype=np.float32)
                # if sample sr != engine SR we keep as-is and resample later
                files.append(Sample(data, sr, fn))
                print(f'Loaded {fn} ({data.shape}, {sr} Hz)')
            except Exception as e:
                print(f'Error loading {fn}: {e}', file=sys.stderr)
    if not files:
        print('No samples found in ./sounds', file=sys.stderr)
    return files

# ----------------------------- Sequencer Structures -----------------------

@dataclass
class Step:
    index: int
    sample_idx: int
    on: bool
    prob: float
    semitone: float
    gain: float
    lowpass: float

    def maybe_trigger(self) -> bool:
        return self.on and (random.random() < self.prob)

@dataclass
class Pattern:
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def random(cls, steps=16, sample_count=1):
        p = cls()
        for i in range(steps):
            p.steps.append(Step(
                index=i,
                #sample_idx=random.randrange(max(1, sample_count)),
                sample_idx=2,
                #on=random.choice([True, False]),
                on=1,
                #prob=random.uniform(0.4, 0.95),
                prob=1,
                #semitone=random.uniform(-12, 12),
                semitone=0,
                #gain=random.uniform(0.5, 1.0),
                gain=1,
                #lowpass=random.uniform(1000.0, 12000.0)
                lowpass=20000.0
            ))
        return p

@dataclass
class Track:
    id: int
    channel: int
    pattern: Pattern

@dataclass
class PlayEvent:
    buffer: np.ndarray  # mono 1D buffer
    channel: int
    pos: int = 0

# ----------------------------- Audio Engine ------------------------------

class AudioEngine:
    def __init__(self, sr=44100, blocksize=512, channels_out=4):
        self.sr = sr
        self.blocksize = blocksize
        self.channels_out = channels_out
        self.active_buffers: List[PlayEvent] = []
        self.lock = threading.Lock()
        self.stream = None
        self.running = False

    def start(self):
        self.stream = sd.OutputStream(
            samplerate=self.sr, blocksize=self.blocksize,
            channels=self.channels_out, dtype='float32',
            latency=LATENCY, callback=self.callback
        )
        self.stream.start()
        self.running = True
        print('Audio stream started.')

    def callback(self, outdata, frames, time_info, status):
        # outdata shape: (frames, channels_out)
        out = np.zeros((frames, self.channels_out), dtype=np.float32)
        with self.lock:
            # mix active buffers
            for ev in self.active_buffers:
                remaining = ev.buffer.shape[0] - ev.pos
                if remaining <= 0:
                    continue
                n = min(frames, remaining)
                # out[:n, ev.channel] is shape (n,)
                slice_buf = ev.buffer[ev.pos:ev.pos + n]
                # ensure slice_buf shape matches (n,)
                if slice_buf.shape[0] != n:
                    slice_buf = np.resize(slice_buf, (n,))
                out[:n, ev.channel] += slice_buf
                ev.pos += n
            # remove finished buffers by rebuilding list (avoid numpy __eq__ issues)
            self.active_buffers = [ev for ev in self.active_buffers if ev.pos < ev.buffer.shape[0]]
        # soft clip and write
        np.clip(out, -1.0, 1.0, out)
        outdata[:] = out

    def play_buffer(self, buf: np.ndarray, channel: int):
        # ensure mono 1D
        if buf.ndim > 1:
            buf = np.mean(buf, axis=1)
        # convert to float32
        buf = np.asarray(buf, dtype=np.float32)
        with self.lock:
            self.active_buffers.append(PlayEvent(buf, channel))

# ----------------------------- Sequencer Core ----------------------------

class Sequencer:
    def __init__(self, engine: AudioEngine, samples: List[Sample]):
        self.engine = engine
        self.samples = samples
        self.tracks: List[Track] = [
            Track(i, TRACK_TO_CHANNEL[i % len(TRACK_TO_CHANNEL)], Pattern.random(STEPS, len(samples)))
            for i in range(TRACKS)
        ]
        self.loop_repeats = random.randint(REPEAT_MIN, REPEAT_MAX)

    def regenerate_patterns(self):
        for t in self.tracks:
            t.pattern = Pattern.random(STEPS, len(self.samples))
        self.loop_repeats = random.randint(REPEAT_MIN, REPEAT_MAX)
        print(f'Generated new patterns (repeats {self.loop_repeats})')

    async def run(self):
        step_duration = 60.0 / BPM / 4.0
        loops = 0
        while True:
            for step_idx in range(STEPS):
                for track in self.tracks:
                    step = track.pattern.steps[step_idx]
                    if step.maybe_trigger():
                        # guard samples list
                        if not self.samples:
                            continue
                        sample = self.samples[step.sample_idx % len(self.samples)]
                        # if sample SR differs, resample naive
                        ratio = semitone_to_ratio(step.semitone)
                        sig = sample.data
                        if sample.sr != SR:
                            # naive resample to engine SR by linear interpolation
                            # compute ratio factor = sample_sr / SR, then apply resample_linear accordingly
                            res_ratio = sample.sr / SR
                            sig = resample_linear(sig, res_ratio)
                        # apply pitch by resampling according to semitone ratio
                        sig = resample_linear(sig, ratio)
                        sig = apply_gain(sig, step.gain)
                        sig = lowpass_4pole(sig, step.lowpass, SR)
                        self.engine.play_buffer(sig, track.channel)
                await asyncio.sleep(step_duration)
            loops += 1
            if loops >= self.loop_repeats:
                self.regenerate_patterns()
                loops = 0

# ----------------------------- Main --------------------------------------

def main():
    samples = load_samples(SOUNDS_DIR)
    if not samples:
        print("No samples loaded. Put sample files into ./sounds and restart.")
        return
    # detect default device channels optionally (graceful fallback)
    try:
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        info = sd.query_devices(default_out, 'output')
        ch = info['max_output_channels']
        print(f'Default output device: {info["name"]} with {ch} channels.')
    except Exception:
        ch = TRACKS
    channels_out = max(ch, TRACKS)
    engine = AudioEngine(SR, BLOCKSIZE, channels_out)
    try:
        engine.start()
    except Exception as e:
        print('Failed to start audio engine:', e, file=sys.stderr)
        return
    seq = Sequencer(engine, samples)
    try:
        asyncio.run(seq.run())
    except KeyboardInterrupt:
        print('Stopped by user.')
    finally:
        if engine.stream:
            try:
                engine.stream.stop()
                engine.stream.close()
            except Exception:
                pass

if __name__ == '__main__':
    main()

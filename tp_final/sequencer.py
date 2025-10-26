import os
import sys
import time
import math
import random
import asyncio
import threading
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import soundfile as sf
import sounddevice as sd

# ----------------------------- Configuration -----------------------------
SOUNDS_DIR = './sounds'
BPM = 100
STEPS = 16
TRACKS = 1
# How many times the current 16-step sequence repeats before regenerating (random 2..8)
REPEAT_MIN = 2
REPEAT_MAX = 8

# Audio settings - try to detect device channels; default to 4 outputs
SR = 44100  # sample rate
BLOCKSIZE = 2048
LATENCY = 'low'

# Map each track to an output channel index (0-based). If your interface has multiple
# output channels, adjust these. UMC404HD usually exposes 4 outputs => channels 0..3
TRACK_TO_CHANNEL = list(range(TRACKS))

# FX defaults
DEFAULT_DELAY_TIME = 0.25  # seconds
DEFAULT_DELAY_FEEDBACK = 0.25
DEFAULT_LOWPASS_CUTOFF = 8000.0  # Hz

# ----------------------------- Helper DSP --------------------------------

def semitone_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)

def resample_linear(signal: np.ndarray, ratio: float) -> np.ndarray:
    # Very simple linear resampling. signal shape (n,) or (n, channels)
    if ratio == 1.0:
        return signal
    n = signal.shape[0]
    new_n = int(max(1, math.floor(n / ratio)))
    if signal.ndim == 1:
        old_idx = np.arange(new_n) * ratio
        i0 = np.floor(old_idx).astype(int)
        frac = old_idx - i0
        i1 = np.minimum(i0 + 1, n - 1)
        return (1 - frac) * signal[i0] + frac * signal[i1]
    else:
        channels = signal.shape[1]
        out = np.zeros((new_n, channels), dtype=signal.dtype)
        old_idx = np.arange(new_n) * ratio
        i0 = np.floor(old_idx).astype(int)
        frac = old_idx - i0
        i1 = np.minimum(i0 + 1, n - 1)
        for c in range(channels):
            out[:, c] = (1 - frac) * signal[i0, c] + frac * signal[i1, c]
        return out

def apply_gain(signal: np.ndarray, gain: float) -> np.ndarray:
    return signal * gain

def apply_pan(signal: np.ndarray, pan: float) -> np.ndarray:
    # pan: -1 (left) .. 1 (right); for mono input produce stereo pair
    if signal.ndim == 1:
        left = math.cos((pan + 1) * math.pi/4)  # equal-power panning
        right = math.sin((pan + 1) * math.pi/4)
        return np.column_stack((signal * left, signal * right))
    else:
        # if stereo, apply simple pan by scaling channels
        left = math.cos((pan + 1) * math.pi/4)
        right = math.sin((pan + 1) * math.pi/4)
        return signal * np.array([left, right])

def one_pole_lowpass(signal: np.ndarray, cutoff: float, sr: int):
    # simple 1-pole IIR for each channel
    dt = 1.0/sr
    rc = 1.0/(2 * math.pi * max(1.0, cutoff))
    alpha = dt / (rc + dt)
    if signal.ndim == 1:
        out = np.zeros_like(signal)
        y = 0.0
        for i, x in enumerate(signal):
            y = y + alpha * (x - y)
            out[i] = y
        return out
    else:
        out = np.zeros_like(signal)
        channels = signal.shape[1]
        y = np.zeros(channels)
        for i in range(signal.shape[0]):
            x = signal[i]
            y = y + alpha * (x - y)
            out[i] = y
        return out

# ----------------------------- Audio Loader -------------------------------

@dataclass
class Sample:
    data: np.ndarray  # shape (n,) or (n,2)
    sr: int
    name: str

def load_samples(folder: str) -> List[Sample]:
    files = []
    if not os.path.isdir(folder):
        print(f"Sounds folder '{folder}' not found.", file=sys.stderr)
        return files
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if os.path.isfile(p) and fn.lower().endswith(('.wav', '.flac', '.aiff', '.aif')):
            try:
                data, sr = sf.read(p, always_2d=False)
                # Convert integers to float32 and normalize if needed
                data = np.asarray(data, dtype=np.float32)
                # If sample rate differs from engine SR, resample now (simple)
                if sr != SR:
                    # naive resample using linear interpolation
                    ratio = sr / SR
                    data = resample_linear(data, ratio)
                    sr = SR
                files.append(Sample(data, sr, fn))
                print(f'Loaded {fn} ({data.shape}, {sr} Hz)')
            except Exception as e:
                print(f'Failed to load {p}: {e}', file=sys.stderr)
    if not files:
        print('No samples found in ./sounds. Please add some WAV/FLAC files.', file=sys.stderr)
    return files

# ----------------------------- Sequencer Objects --------------------------

@dataclass
class Step:
    index: int
    sample_idx: int
    prob: float
    semitone: float
    gain: float
    pan: float
    lowpass: float
    delay_time: float
    delay_feedback: float

    def maybe_trigger(self) -> bool:
        return random.random() < self.prob

@dataclass
class Pattern:
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def random(cls, steps=16, sample_count=1):
        s = cls()
        for i in range(steps):
            step = Step(
                index=i,
                sample_idx=random.randrange(sample_count),
                prob=random.uniform(0.4, 0.95),
                semitone=random.uniform(-12, 12),
                gain=random.uniform(0.4, 1.0),
                pan=random.uniform(-1.0, 1.0),
                lowpass=random.uniform(2000.0, 12000.0),
                delay_time=random.uniform(0.0, 0.5),
                delay_feedback=random.uniform(0.0, 0.5)
            )
            s.steps.append(step)
        return s

@dataclass
class Track:
    id: int
    channel: int  # hardware output channel
    pattern: Pattern
    fx: dict = field(default_factory=dict)

# Event to play a rendered buffer to a specific channel
@dataclass
class PlayEvent:
    buffer: np.ndarray  # shape (n, channels_out)
    channel: int
    position: int = 0  # sample position in buffer

# ----------------------------- Audio Engine -------------------------------

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
        self.running = True
        # open output stream with callback
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                blocksize=self.blocksize,
                channels=self.channels_out,
                dtype='float32',
                latency=LATENCY,
                callback=self.callback
            )
            self.stream.start()
            print('Audio stream started.')
        except Exception as e:
            print('Failed to start audio stream:', e, file=sys.stderr)
            raise

    def stop(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

    def callback(self, outdata, frames, time_info, status):
        # outdata shape: (frames, channels_out)
        out = np.zeros((frames, self.channels_out), dtype=np.float32)
        # mix active buffers
        with self.lock:
            remaining = []
            for ev in self.active_buffers:
                buf = ev.buffer
                pos = ev.position
                to_copy = min(frames, buf.shape[0] - pos)
                if to_copy > 0:
                    out[:to_copy, :] += buf[pos:pos+to_copy, :self.channels_out]
                    ev.position += to_copy
                if ev.position < buf.shape[0]:
                    remaining.append(ev)
            self.active_buffers = remaining
        # simple soft clipping
        np.clip(out, -1.0, 1.0, out)
        outdata[:] = out

    def play_buffer(self, buffer: np.ndarray, channel: int):
        # buffer shape expected (n, channels_out) but we'll place audio in the correct channel(s)
        # If buffer is stereo and channels_out >=2, map to pair starting at 'channel' (if possible),
        # else map mono to specific channel index.
        with self.lock:
            # Expand buffer to channels_out if needed
            if buffer.ndim == 1:
                b = np.zeros((buffer.shape[0], self.channels_out), dtype=np.float32)
                b[:, channel] = buffer
            else:
                # buffer has 2 columns (stereo) or more. We'll map stereo to [channel, channel+1] if possible
                if buffer.shape[1] >= 2 and channel <= self.channels_out - 2:
                    b = np.zeros((buffer.shape[0], self.channels_out), dtype=np.float32)
                    b[:, channel] = buffer[:, 0]
                    b[:, channel+1] = buffer[:, 1]
                else:
                    b = np.zeros((buffer.shape[0], self.channels_out), dtype=np.float32)
                    # if stereo but only one channel mapping, average to mono
                    mono = buffer.mean(axis=1)
                    b[:, channel] = mono
            ev = PlayEvent(b, channel, 0)
            self.active_buffers.append(ev)

# ----------------------------- Sequencer / Render -------------------------

class Sequencer:
    def __init__(self, samples: List[Sample], tracks=4, steps=16, bpm=120, engine: Optional[AudioEngine]=None):
        self.samples = samples
        self.tracks_count = tracks
        self.steps = steps
        self.bpm = bpm
        self.engine = engine
        self.tracks: List[Track] = []
        self._stop_flag = False
        self.current_repeat = 0
        self.repeat_target = random.randint(REPEAT_MIN, REPEAT_MAX)
        self._build_tracks()

    def _build_tracks(self):
        sc = max(1, len(self.samples))
        self.tracks = []
        for t in range(self.tracks_count):
            pattern = Pattern.random(self.steps, sc)
            fx = {
                'lowpass_cutoff': DEFAULT_LOWPASS_CUTOFF,
                'delay_time': DEFAULT_DELAY_TIME,
                'delay_feedback': DEFAULT_DELAY_FEEDBACK
            }
            track = Track(t, TRACK_TO_CHANNEL[t], pattern, fx)
            self.tracks.append(track)
        print(f'Built {len(self.tracks)} tracks, each with {self.steps} steps. Repeat target: {self.repeat_target}')

    def regenerate(self):
        self.repeat_target = random.randint(REPEAT_MIN, REPEAT_MAX)
        self._build_tracks()
        self.current_repeat = 0
        print('Regenerated patterns. New repeat target:', self.repeat_target)

    def render_step_to_buffer(self, sample: Sample, step: Step) -> np.ndarray:
        # Render the sample with step parameters and returned mixed buffer (n, channels_out)
        data = sample.data
        # if sample sr different handled earlier; assume sample.sr == SR
        # pitch via resampling: ratio = original_sr / (sr * pitch_ratio) ; but easier: ratio = semitone_ratio
        ratio = semitone_to_ratio(step.semitone)
        # To shift pitch up, we need to shorten (play faster), so resample with ratio (orig_len / ratio?):
        # If semitone > 0, ratio>1 => speed up -> new length = old / ratio. So use ratio in resample function as factor.
        data_rs = resample_linear(data, ratio)
        # Ensure mono
        if data_rs.ndim == 2 and data_rs.shape[1] == 2:
            # keep stereo for panning later
            pass
        elif data_rs.ndim == 2:
            data_rs = data_rs[:, 0]
        # Apply gain
        data_g = apply_gain(data_rs, step.gain)
        # Apply pan -> get stereo
        #data_p = apply_pan(data_g, step.pan)
        data_p = apply_pan(data_g, step.)
        # Lowpass
        lp = one_pole_lowpass(data_p, step.lowpass, SR)
        if 1:
        # Delay effect (per-track simple feedback delay implemented by mixing delayed copy)
        # if step.delay_time > 0.001 and step.delay_feedback > 0.001:
        #     delay_samples = int(step.delay_time * SR)
        #     # build buffer with extra space for echoes
        #     max_echo = 4
        #     out_len = lp.shape[0] + delay_samples * max_echo
        #     out = np.zeros((out_len, lp.shape[1]), dtype=np.float32)
        #     out[:lp.shape[0], :] += lp
        #     for e in range(1, max_echo+1):
        #         start = e * delay_samples
        #         amp = (step.delay_feedback ** e)
        #         if start < out_len:
        #             end = start + lp.shape[0]
        #             if end > out_len:
        #                 # truncate
        #                 end = out_len
        #                 length = end - start
        #                 out[start:end, :] += lp[:length, :] * amp
        #             else:
        #                 out[start:end, :] += lp * amp
        #     # Trim silence tail if too long (limit to 6 seconds)
        #     max_len = SR * 6
        #     if out.shape[0] > max_len:
        #         out = out[:max_len, :]
        #     return out
            # pad to stereo with channels_out later
            if lp.ndim == 1:
                # convert to stereo
                lp = np.column_stack((lp, lp))
            return lp

    async def start(self):
        beat_interval = 60.0 / self.bpm  # quarter note
        step_interval = beat_interval / 4.0  # 16th notes
        step_index = 0
        print(f'Sequencer started at {self.bpm} BPM ({step_interval:.4f}s per 16th).')
        while not self._stop_flag:
            t0 = time.time()
            # For each track, evaluate its step
            for tr in self.tracks:
                step = tr.pattern.steps[step_index % self.steps]
                if step.maybe_trigger():
                    # render buffer and send to engine
                    # sample selection may be out of range if no samples; handle
                    if self.samples:
                        sample = self.samples[step.sample_idx % len(self.samples)]
                        buf = self.render_step_to_buffer(sample, step)
                        # play on engine mapped channel
                        if self.engine:
                            self.engine.play_buffer(buf, tr.channel)
            step_index = (step_index + 1) % self.steps
            # if we've completed a full cycle (step_index==0), increment repeat counter
            if step_index == 0:
                self.current_repeat += 1
                if self.current_repeat >= self.repeat_target:
                    self.regenerate()
            # sleep until next step, adjusted for drift
            t1 = time.time()
            elapsed = t1 - t0
            to_sleep = step_interval - elapsed
            if to_sleep > 0:
                await asyncio.sleep(to_sleep)
            else:
                # we're running late; yield control briefly
                await asyncio.sleep(0)

    def stop(self):
        self._stop_flag = True

# ----------------------------- Main --------------------------------------

def main():
    print('Starting Python 4-track probabilistic sequencer.')
    samples = load_samples(SOUNDS_DIR)
    if not samples:
        print('No samples loaded; exiting. Put sample files in ./sounds and restart.')
        return
    # Try to detect default output device and available channels
    try:
        default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        info = sd.query_devices(default_out, 'output')
        ch = info['max_output_channels']
        print(f'Default output device: {info["name"]} with {ch} channels.')
    except Exception as e:
        print('Could not query default output device:', e, file=sys.stderr)
        ch = max(TRACKS, 2)

    channels_out = max(ch, TRACKS)
    engine = AudioEngine(sr=SR, blocksize=BLOCKSIZE, channels_out=channels_out)
    try:
        engine.start()
    except Exception as e:
        print('Audio engine failed to start. Exiting.')
        return

    seq = Sequencer(samples, tracks=TRACKS, steps=STEPS, bpm=BPM, engine=engine)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(seq.start())
    except KeyboardInterrupt:
        print('Stopping sequencer...')
    finally:
        seq.stop()
        engine.stop()

if __name__ == '__main__':
    main()

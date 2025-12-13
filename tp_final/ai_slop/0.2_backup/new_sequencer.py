#!/usr/bin/env python3
"""
Deterministic Microsample Sequencer
- Clock / Sequencer / AudioEngine separation
- Per-run logging (runs/run_YYYYMMDD_HHMMSS.json)
- Per-track stacks with memory decay
- Density drift & sample-entropy based intensity nudging
- Deterministic sample loading (no ORDER BY RANDOM)
"""

import os
import sys
import math
import random
import threading
import argparse
import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import deque, Counter

import numpy as np
import soundfile as sf
import sounddevice as sd

# ---------------- Configuration via Arguments ----------------
def parse_arguments():
    parser = argparse.ArgumentParser(description='Microsample Sequencer')

    # Audio configuration
    parser.add_argument('--db-path', default='microsamples.db', help='SQLite database path')
    parser.add_argument('--samples-dir', default='samples', help='Directory containing sample files')
    parser.add_argument('--bpm', type=float, default=120, help='Tempo in BPM')
    parser.add_argument('--steps', type=int, default=16, help='Steps per pattern')
    parser.add_argument('--channels', type=int, choices=[1, 2, 4], default=1,
                        help='Output channels: 1=mono, 2=stereo, 4=quadraphonic')
    parser.add_argument('--sr', type=int, default=44100, help='Sample rate')
    parser.add_argument('--blocksize', type=int, default=4096, help='Audio block size')
    parser.add_argument('--latency', default='high', help='Audio latency setting')

    # Sequence configuration
    parser.add_argument('--repeat-min', type=int, default=4, help='Minimum pattern repetitions')
    parser.add_argument('--repeat-max', type=int, default=16, help='Maximum pattern repetitions')
    parser.add_argument('--samples-per-track', type=int, default=1000,
                        help='Number of samples to fetch per track type')

    # Sample processing / fades
    parser.add_argument('--fade-duration', type=float, default=0.005,
                        help='Fade in/out duration in seconds (default: 5ms)')

    # Envelope (ADSR) in seconds
    parser.add_argument('--env-attack', type=float, default=0.001, help='Envelope attack (s)')
    parser.add_argument('--env-decay', type=float, default=1, help='Envelope decay (s)')
    parser.add_argument('--env-sustain', type=float, default=1, help='Envelope sustain level (0..1)')
    parser.add_argument('--env-release', type=float, default=1, help='Envelope release (s)')

    # Reverse play probability per step
    parser.add_argument('--reverse-prob', type=float, default=0.2, help='Per-step probability to play sample reversed')

    # Stack options
    parser.add_argument('--stack', action='store_true', help='Enable played-samples stack mode')
    parser.add_argument('--stack-prob', type=float, default=0.05, help='When stack enabled, probability to pick sample from stack for a step')
    parser.add_argument('--stack-max', type=int, default=64, help='Maximum size of the played-samples stack')
    parser.add_argument('--stack-decay', type=int, default=128, help='How many steps before a played sample is expired from memory')

    # Soft clipping
    parser.add_argument('--clip-drive', type=float, default=1.0, help='Soft-clip drive (higher = harder saturation)')
    parser.add_argument('--clip-asym', type=float, default=-0.3, help='Asymmetry -1.0..1.0 where positive biases positive side (tube-like)')
    parser.add_argument('--clip-makeup', type=float, default=3, help='Makeup gain applied prior to soft clip')

    # Random control
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (default: random)')

    # Artistic options
    parser.add_argument('--enable-density-drift', action='store_true', help='Enable slow density drift over time')
    parser.add_argument('--density-drift-rate', type=float, default=0.1, help='Maximum density drift step per regeneration')
    parser.add_argument('--entropy-nudge', action='store_true', help='Enable sample-entropy based intensity nudging')

    return parser.parse_args()

# ---------------- Channel Mapping ----------------
def get_channel_mapping(output_channels: int) -> Dict[str, int]:
    """Map stem types to output channels based on channel configuration"""
    if output_channels == 1:
        return {'drums': 0, 'bass': 0, 'other': 0, 'vocals': 0}
    elif output_channels == 2:
        return {'drums': 0, 'bass': 0, 'other': 1, 'vocals': 1}
    else:  # 4 channels
        return {'drums': 0, 'bass': 1, 'other': 2, 'vocals': 3}

# ---------------- DSP helpers -------------------
def semitone_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)

def resample_linear(signal: np.ndarray, ratio: float) -> np.ndarray:
    """Fast linear resampling"""
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

def apply_fade(signal: np.ndarray, fade_samples: int, sample_rate: int) -> np.ndarray:
    """Apply fade in and fade out to prevent clicks"""
    if len(signal) == 0:
        return signal
    if len(signal) <= fade_samples * 2:
        fade_len = min(fade_samples, len(signal) // 2)
        if fade_len > 0:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)
            if signal.ndim == 1:
                signal[:fade_len] *= fade_in
                signal[-fade_len:] *= fade_out
            else:
                for c in range(signal.shape[1]):
                    signal[:fade_len, c] *= fade_in
                    signal[-fade_len:, c] *= fade_out
    else:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        if signal.ndim == 1:
            signal[:fade_samples] *= fade_in
            signal[-fade_samples:] *= fade_out
        else:
            for c in range(signal.shape[1]):
                signal[:fade_samples, c] *= fade_in
                signal[-fade_samples:, c] *= fade_out
    return signal

def lowpass_4pole(signal: np.ndarray, cutoff: float, sr: int) -> np.ndarray:
    """Simple 4-pole lowpass filter"""
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    dt = 1.0 / sr
    rc = 1.0 / (2 * math.pi * max(1.0, cutoff))
    alpha = dt / (rc + dt)
    y1 = y2 = y3 = y4 = 0.0
    out = np.zeros_like(signal, dtype=np.float32)
    for i, x in enumerate(signal):
        y1 += alpha * (x - y1)
        y2 += alpha * (y1 - y2)
        y3 += alpha * (y2 - y3)
        y4 += alpha * (y3 - y4)
        out[i] = y4
    return out

def apply_adsr_envelope(signal: np.ndarray, sr: int, attack: float, decay: float, sustain: float, release: float) -> np.ndarray:
    length = signal.shape[0]
    a = max(0, int(round(attack * sr)))
    d = max(0, int(round(decay * sr)))
    r = max(0, int(round(release * sr)))
    sustain_len = max(0, length - (a + d + r))
    if length < (a + d + r) and length > 0:
        total = a + d + r
        if total > 0:
            scale = length / total
            a = max(0, int(round(a * scale)))
            d = max(0, int(round(d * scale)))
            r = max(0, length - (a + d))
            sustain_len = 0

    env = np.ones(length, dtype=np.float32) * sustain
    pos = 0
    if a > 0:
        env[pos:pos + a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    pos += a
    if d > 0:
        env[pos:pos + d] = np.linspace(1.0, sustain, d, dtype=np.float32)
    pos += d
    pos += sustain_len
    if r > 0:
        env[pos:pos + r] = np.linspace(sustain, 0.0, r, dtype=np.float32)
    return signal * env

def soft_clip_channel(x: np.ndarray, drive: float = 1.0, asym: float = 0.0) -> np.ndarray:
    y = x * drive
    pos_scale = 1.0 + max(0.0, asym)
    neg_scale = 1.0 + max(0.0, -asym)
    pos = np.tanh(pos_scale * np.maximum(0.0, y))
    neg = np.tanh(neg_scale * np.minimum(0.0, y))
    return pos + neg

def apply_soft_clip(out: np.ndarray, drive: float, asym: float, makeup: float):
    if makeup != 1.0:
        out *= makeup
    for ch in range(out.shape[1]):
        out[:, ch] = soft_clip_channel(out[:, ch], drive, asym)
    np.clip(out, -1.0, 1.0, out)
    return out

# ---------------- Sample / DB -----------------
@dataclass
class Sample:
    data: np.ndarray
    sr: int
    name: str
    stem_type: str
    original_position: float
    rms: float
    duration: float

class SampleDatabase:
    def __init__(self, db_path: str, samples_dir: str, rng: random.Random):
        self.db_path = db_path
        self.samples_dir = samples_dir
        self.conn = sqlite3.connect(db_path)
        self.rng = rng

    def get_samples_by_stem_type(self, stem_type: str, limit: int = 100) -> List[Sample]:
        cursor = self.conn.cursor()
        query = """
            SELECT sample_filename, stem_type, sample_position, duration, metadata_json
            FROM samples
            WHERE stem_type = ?
            ORDER BY sample_filename ASC
        """
        cursor.execute(query, (stem_type,))
        rows = cursor.fetchall()
        # deterministic shuffle
        self.rng.shuffle(rows)
        if limit is not None:
            rows = rows[:limit]

        samples: List[Sample] = []
        for row in rows:
            filename, stem_type_row, position, duration, metadata_json = row
            audio_path = os.path.join(self.samples_dir, filename)
            try:
                if os.path.exists(audio_path):
                    data, sr = sf.read(audio_path, always_2d=False)
                    data = np.asarray(data, dtype=np.float32)
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    sample = Sample(
                        data=data,
                        sr=sr,
                        name=f"{metadata.get('artist', 'Unknown')} - {metadata.get('song_title', 'Unknown')} - {stem_type_row}",
                        stem_type=stem_type_row,
                        original_position=position,
                        rms=metadata.get('rms', 0.5),
                        duration=float(duration) if duration is not None else float(len(data) / sr)
                    )
                    samples.append(sample)
                else:
                    print(f"Warning: Sample file not found: {audio_path}")
            except Exception as e:
                print(f"Error loading sample {filename}: {e}")
                continue
        print(f"Loaded {len(samples)} {stem_type} samples from {self.samples_dir}")
        return samples

    def close(self):
        self.conn.close()

def load_samples_from_db(db_path: str, samples_dir: str, samples_per_track: int, rng: random.Random) -> Dict[str, List[Sample]]:
    db = SampleDatabase(db_path, samples_dir, rng)
    stem_types = ['drums', 'bass', 'other', 'vocals']
    samples_by_type: Dict[str, List[Sample]] = {}
    for stem_type in stem_types:
        samples = db.get_samples_by_stem_type(stem_type, samples_per_track)
        samples_by_type[stem_type] = samples
        if not samples:
            print(f"Warning: No {stem_type} samples found in database!")
    db.close()
    return samples_by_type

# ---------------- Sequencer primitives -----------------
@dataclass
class Step:
    index: int
    sample_idx: int
    on: bool
    prob: float
    semitone: float
    gain: float
    lowpass: float

    def maybe_trigger(self, rng: random.Random) -> bool:
        return self.on and (rng.random() < self.prob)

@dataclass
class Pattern:
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def random(cls, steps=16, sample_count=1, rng: Optional[random.Random] = None, pattern_density: float = 0.5, pattern_intensity: float = 1.0):
        rng = rng or random
        p = cls()
        for i in range(steps):
            p.steps.append(Step(
                index=i,
                sample_idx=rng.randrange(max(1, sample_count)),
                on=(rng.random() < pattern_density),
                prob=rng.uniform(0.1, 1.0),
                semitone=0.0,
                gain=rng.uniform(0.1 * pattern_intensity, 0.2 * pattern_intensity),
                lowpass=20000.0
            ))
        return p

@dataclass
class Track:
    id: int
    stem_type: str
    channel: int
    samples: List[Sample]
    pattern: Pattern
    rng: random.Random
    played_stack: deque = field(default_factory=lambda: deque())
    stack_ages: Dict[int, int] = field(default_factory=dict)  # sample index -> age in steps
    stack_lock: threading.Lock = field(default_factory=threading.Lock)
    events: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class PlayEvent:
    buffer: np.ndarray
    channel: int
    pos: int = 0

# ---------------- Clock -----------------
class Clock:
    """Sample-accurate clock. It does NOT make musical decisions; it only provides step boundaries."""
    def __init__(self, bpm: float, sr: int, steps_per_beat: int = 4):
        self.bpm = bpm
        self.sr = sr
        # samples per quarter note = sr * 60 / bpm
        quarter_samples = sr * 60.0 / bpm
        # steps per quarter note default = 4 (16th notes). steps_per_beat = 4 means 16th grid.
        self.samples_per_step = int(round(quarter_samples / steps_per_beat))
        self.sample_pos = 0
        self.next_step_sample = self.samples_per_step

    def advance(self, frames: int) -> List[int]:
        """
        Advance clock by 'frames' samples (inside audio callback).
        Return a list of step indices that occurred during this block (as offsets from current sample_pos).
        We return a list of absolute sample positions (sample_pos where step occurred).
        """
        events = []
        start = self.sample_pos
        end = self.sample_pos + frames
        while self.next_step_sample <= end:
            events.append(self.next_step_sample)
            self.next_step_sample += self.samples_per_step
        self.sample_pos = end
        return events

    def reset(self):
        self.sample_pos = 0
        self.next_step_sample = self.samples_per_step

# ---------------- Sequencer -----------------
class Sequencer:
    def __init__(self, samples_by_type: Dict[str, List[Sample]], config: argparse.Namespace, rng: random.Random, run_logger: "RunLogger"):
        self.samples_by_type = samples_by_type
        self.config = config
        self.rng = rng
        self.bpm = config.bpm
        self.steps = config.steps
        self.fade_samples = int(config.fade_duration * config.sr)
        self.env_attack = config.env_attack
        self.env_decay = config.env_decay
        self.env_sustain = config.env_sustain
        self.env_release = config.env_release
        self.reverse_prob = max(0.0, min(1.0, config.reverse_prob))

        self.stack_enabled = bool(config.stack)
        self.stack_prob = config.stack_prob
        self.stack_max = max(1, config.stack_max)
        self.stack_decay_steps = max(1, config.stack_decay)

        # logging
        self.run_logger = run_logger
        self.run_meta: Dict[str, Any] = {
            "seed": config.seed,
            "bpm": self.bpm,
            "steps": self.steps,
            "stack_enabled": self.stack_enabled,
            "stack_prob": self.stack_prob,
            "stack_max": self.stack_max,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # seeded shared global density & intensity
        self.density = rng.random()
        self.intensity = rng.uniform(1.0, 5.0)

        # build tracks
        self.channel_map = get_channel_mapping(config.channels)
        self.tracks: List[Track] = []
        stem_types = ['drums', 'bass', 'other', 'vocals']
        for i, stem_type in enumerate(stem_types):
            samples = samples_by_type.get(stem_type, [])
            track_rng = random.Random(self.rng.randint(0, 2**30))
            pat = Pattern.random(self.steps, max(1, len(samples)), rng=track_rng, pattern_density=self.density, pattern_intensity=self.intensity)
            track = Track(
                id=i,
                stem_type=stem_type,
                channel=self.channel_map[stem_type],
                samples=samples,
                pattern=pat,
                rng=track_rng
            )
            # ensure each track has its bounded deque structure for stack (store sample indices)
            track.played_stack = deque(maxlen=self.stack_max)
            track.stack_ages = {}
            track.events = []
            self.tracks.append(track)

        # loop control
        self.loop_count = 0
        self.loop_target = self.rng.randint(config.repeat_min, config.repeat_max)

        # logging accumulator
        # (No longer needed: patterns/events are owned by RunLogger)

        # Start the first pattern block in the run log (metadata FIRST, then events)
        self.run_logger.start_pattern(self, reason="init")

    # ---------- decision logic ----------
    def decide_for_step(self, step_index: int) -> List[Tuple[Track, Sample, bool, bool, Step]]:
        """
        For a global step index, decide which tracks will trigger and which sample is chosen.
        Returns a list of tuples: (track, chosen_sample, used_stack_bool, reversed_bool, step_obj)
        """
        results: List[Tuple[Track, Sample, bool, bool, Step]] = []
        for track in self.tracks:
            step = track.pattern.steps[step_index]
            if not (step.on and (track.rng.random() < step.prob)):
                continue  # not triggered
            if not track.samples:
                continue
            use_stack_choice = False
            chosen_sample: Optional[Sample] = None

            # Attempt stack pick
            if self.stack_enabled and (track.rng.random() < self.stack_prob) and len(track.played_stack) > 0:
                # pick by recency weight (younger = higher weight)
                with track.stack_lock:
                    # compute weights: newer elements are at the right (deque append at right)
                    n = len(track.played_stack)
                    # recency index: 0..n-1, 0 = oldest, n-1 = newest
                    # Let's weight by (index + 1) so newest have high weight
                    weights = [(idx + 1) for idx in range(n)]
                    total = sum(weights)
                    r = track.rng.random() * total
                    acc = 0.0
                    chosen_idx = 0
                    for idx, w in enumerate(weights):
                        acc += w
                        if r <= acc:
                            chosen_idx = idx
                            break
                    # chosen_idx corresponds to left-to-right: oldest..newest
                    # map to deque element
                    chosen_sample_idx = list(track.played_stack)[chosen_idx]
                    chosen_sample = track.samples[chosen_sample_idx]
                    # remove from deque (we're consuming memory)
                    try:
                        track.played_stack.remove(chosen_sample_idx)
                        # also remove age entry
                        track.stack_ages.pop(chosen_sample_idx, None)
                    except ValueError:
                        # fallback: ignore
                        pass
                    use_stack_choice = True

            # If not chosen from stack, pick from sample list
            if not use_stack_choice:
                samp_idx = step.sample_idx % len(track.samples)
                chosen_sample = track.samples[samp_idx]

            # chance to reverse
            reversed_play = (track.rng.random() < self.reverse_prob)

            results.append((track, chosen_sample, use_stack_choice, reversed_play, step))
            # Update stack aging bookkeeping after decision (we'll append later if needed)

        return results

    def commit_played_sample_to_stack(self, track: Track, chosen_sample: Sample, use_stack_choice: bool):
        """Append sample index into track.played_stack if not taken from stack and not already present.
        Use identity by index (store integer sample index) for lightweight comparisons.
        """
        if not self.stack_enabled:
            return
        if use_stack_choice:
            return
        # find index in track.samples
        try:
            idx = track.samples.index(chosen_sample)
        except ValueError:
            return
        with track.stack_lock:
            if idx not in track.played_stack:
                track.played_stack.append(idx)
                track.stack_ages[idx] = 0  # reset age

            # enforce maxlen already handled by deque
            while len(track.played_stack) > self.stack_max:
                old_idx = track.played_stack.popleft()
                track.stack_ages.pop(old_idx, None)

    def age_and_decay_stacks(self):
        """Increment ages, and purge entries older than stack_decay_steps."""
        for track in self.tracks:
            with track.stack_lock:
                to_remove = []
                for idx in list(track.stack_ages.keys()):
                    track.stack_ages[idx] += 1
                    if track.stack_ages[idx] > self.stack_decay_steps:
                        to_remove.append(idx)
                for idx in to_remove:
                    try:
                        track.played_stack.remove(idx)
                    except ValueError:
                        pass
                    track.stack_ages.pop(idx, None)

    def record_event(self, step_sample_pos: int, step_index: int, track: Track, sample: Sample, used_stack: bool, reversed_play: bool, step_obj: Step):
        ev = {
            "sample_pos": int(step_sample_pos),
            "step_index": int(step_index),
            "track_id": int(track.id),
            "track_type": track.stem_type,
            "sample_name": sample.name,
            "gain": float(step_obj.gain),
            "semitone": float(step_obj.semitone),
            "lowpass": float(step_obj.lowpass),
            "prob": float(step_obj.prob),
            "used_stack": bool(used_stack),
            "reversed": bool(reversed_play),
            "density": float(self.density),
            "intensity": float(self.intensity),
            "time": time.time()
        }
        track.events.append(ev)

        # Log under the currently active pattern object
        self.run_logger.log_event(ev)

    def regenerate_patterns(self):
        """Regenerate patterns for all tracks; called when loop finishes."""
        # optionally drift density
        if self.config.enable_density_drift:
            delta = self.rng.uniform(-self.config.density_drift_rate, self.config.density_drift_rate)
            self.density = min(max(0.01, self.density + delta), 0.99)
        else:
            self.density = self.density  # unchanged

        # intensity variation
        self.intensity = self.rng.uniform(1.0, 5.0)

        # enforce entropy-driven nudging
        if self.config.entropy_nudge:
            for track in self.tracks:
                # compute entropy of last N events
                N = min(64, len(track.events))
                if N < 8:
                    continue
                last_names = [e["sample_name"] for e in track.events[-N:]]
                counts = Counter(last_names)
                probs = np.array(list(counts.values()), dtype=float) / float(N)
                entropy = -np.sum(probs * np.log2(probs + 1e-12))
                # low entropy -> increase intensity a bit
                if entropy < 2.5:
                    self.intensity = min(6.0, self.intensity + 0.5)

        for track in self.tracks:
            track.pattern = Pattern.random(self.steps, max(1, len(track.samples)), rng=track.rng, pattern_density=self.density, pattern_intensity=self.intensity)

        # choose new loop length
        self.loop_target = self.rng.randint(self.config.repeat_min, self.config.repeat_max)
        print(f"[Regenerated patterns. density={self.density:.3f}, intensity={self.intensity:.3f}, next_repeat={self.loop_target}]")

        # logging: start a new pattern object and snapshot its metadata BEFORE any events occur
        self.run_logger.start_pattern(self, reason="regenerate")

# ---------------- Audio Engine -----------------
class AudioEngine:
    def __init__(self, sequencer: Sequencer, config: argparse.Namespace, rng: random.Random):
        self.seq = sequencer
        self.config = config
        self.rng = rng
        self.sr = config.sr
        self.blocksize = config.blocksize
        self.channels_out = config.channels
        self.fade_samples = int(config.fade_duration * self.sr)
        self.clip_drive = config.clip_drive
        self.clip_asym = config.clip_asym
        self.clip_makeup = config.clip_makeup

        # clock
        self.clock = Clock(self.seq.bpm, self.sr, steps_per_beat=4)

        # sample-accurate scheduling queue: list of PlayEvent
        self.active_events: List[PlayEvent] = []
        self.events_lock = threading.Lock()

        # sounddevice stream
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.channels_out,
            dtype='float32',
            latency=self.config.latency,
            callback=self.callback
        )
        self.running = False

    def start(self):
        self.stream.start()
        self.running = True
        print("Audio engine started.")

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.running = False

    def schedule_play(self, sample: Sample, step_sample_pos: int, channel: int, reversed_play: bool, step: Step):
        """Schedule a sample to be played starting at the next step boundary.
        We compute offset relative to current clock.sample_pos to align playback.
        """
        # prepare signal
        sig = sample.data.copy()
        if reversed_play:
            sig = sig[::-1]
        if sample.sr != self.sr:
            sig = resample_linear(sig, sample.sr / self.sr)
        pitch_ratio = semitone_to_ratio(step.semitone)
        if pitch_ratio != 1.0:
            sig = resample_linear(sig, pitch_ratio)
        sig = apply_gain(sig, step.gain)
        sig = lowpass_4pole(sig, step.lowpass, self.sr)
        if sig.ndim > 1:
            sig = np.mean(sig, axis=1)
        sig = apply_adsr_envelope(sig, self.sr, self.seq.env_attack, self.seq.env_decay, self.seq.env_sustain, self.seq.env_release)
        sig = apply_fade(sig, self.fade_samples, self.sr)

        ev = PlayEvent(np.asarray(sig, dtype=np.float32), channel)
        with self.events_lock:
            self.active_events.append(ev)

    def _mix_active_into(self, out: np.ndarray, start: int, end: int):
        length = end - start
        with self.events_lock:
            # iterate over a copy to allow modification
            for ev in list(self.active_events):
                remaining = ev.buffer.shape[0] - ev.pos
                if remaining <= 0:
                    # remove finished event
                    try:
                        self.active_events.remove(ev)
                    except ValueError:
                        pass
                    continue
                n = min(length, remaining)
                if ev.channel < out.shape[1]:
                    out[start:start + n, ev.channel] += ev.buffer[ev.pos:ev.pos + n]
                ev.pos += n

    def callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, self.channels_out), dtype=np.float32)

        # advance clock and collect step boundaries (absolute sample positions relative to engine.clock.sample_pos)
        step_positions = self.clock.advance(frames)

        # mix and call sequencer when steps occur
        local_cursor = 0
        for pos in step_positions:
            # pos is an absolute sample index (relative to engine.clock), translate to local block offset
            block_offset = pos - (self.clock.sample_pos - frames)
            block_offset = int(block_offset)
            if block_offset > local_cursor and block_offset <= frames:
                self._mix_active_into(out, local_cursor, min(block_offset, frames))
                local_cursor = block_offset

            # compute the logical step index (mod sequencer.steps)
            # step index increments each step; we can compute as (global_step_counter % steps)
            # Here we derive step_index by dividing pos by samples_per_step
            step_index = ((pos - self.clock.samples_per_step) // self.clock.samples_per_step) % self.seq.steps
            # let sequencer decide events for this step
            decisions = self.seq.decide_for_step(step_index)
            for (track, sample, used_stack, reversed_play, step_obj) in decisions:
                # record event in sequencer logs
                self.seq.record_event(pos, step_index, track, sample, used_stack, reversed_play, step_obj)
                # audio schedule
                self.schedule_play(sample, pos, track.channel, reversed_play, step_obj)
                # commit to stack if needed
                self.seq.commit_played_sample_to_stack(track, sample, used_stack)

            # aging stacks on each step
            self.seq.age_and_decay_stacks()

            # loop bookkeeping
            if step_index == (self.seq.steps - 1):
                self.seq.loop_count += 1
                if self.seq.loop_count >= self.seq.loop_target:
                    self.seq.loop_count = 0
                    self.seq.regenerate_patterns()

        # mix remainder of block
        if local_cursor < frames:
            self._mix_active_into(out, local_cursor, frames)

        # soft clip & output
        apply_soft_clip(out, self.clip_drive, self.clip_asym, self.clip_makeup)
        outdata[:] = out

# ---------------- Run logging -----------------
class RunLogger:
    """
    Per-run logger that writes a single JSON file.
    Structure:
      {
        "run_meta": {...},
        "patterns": [
          {
            "pattern_index": 0,
            "reason": "init" | "regenerate",
            "created_at": "...",
            "density": ...,
            "intensity": ...,
            "loop_target": ...,
            "tracks": [...pattern definitions...],
            "events": [...]
          },
          ...
        ]
      }
    """
    def __init__(self, run_meta: Dict[str, Any], runs_dir: Path = Path("runs")):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_path = self.runs_dir / f"run_{ts}.json"

        # Ensure timestamp is present and consistent
        self.run_meta = dict(run_meta)
        self.run_meta.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        self.patterns: List[Dict[str, Any]] = []
        self._current_pattern: Optional[Dict[str, Any]] = None
        self._pattern_index = 0

    def _snapshot_tracks_pattern(self, sequencer: "Sequencer") -> List[Dict[str, Any]]:
        tracks_meta: List[Dict[str, Any]] = []
        for t in sequencer.tracks:
            steps_meta = []
            for s in t.pattern.steps:
                steps_meta.append({
                    "index": int(s.index),
                    "sample_idx": int(s.sample_idx),
                    "on": bool(s.on),
                    "prob": float(s.prob),
                    "semitone": float(s.semitone),
                    "gain": float(s.gain),
                    "lowpass": float(s.lowpass),
                })
            tracks_meta.append({
                "track_id": int(t.id),
                "stem_type": t.stem_type,
                "channel": int(t.channel),
                "sample_count": int(len(t.samples)),
                "steps": steps_meta,
            })
        return tracks_meta

    def start_pattern(self, sequencer: "Sequencer", reason: str):
        # Close previous pattern (optional end marker)
        if self._current_pattern is not None:
            self._current_pattern["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        pat = {
            "pattern_index": int(self._pattern_index),
            "reason": str(reason),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "density": float(sequencer.density),
            "intensity": float(sequencer.intensity),
            "loop_target": int(sequencer.loop_target),
            "tracks": self._snapshot_tracks_pattern(sequencer),
            "events": []
        }
        self.patterns.append(pat)
        self._current_pattern = pat
        self._pattern_index += 1

    def log_event(self, ev: Dict[str, Any]):
        # If for any reason events arrive before pattern init, create a fallback pattern.
        if self._current_pattern is None:
            self.patterns.append({
                "pattern_index": int(self._pattern_index),
                "reason": "implicit_init",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "density": None,
                "intensity": None,
                "loop_target": None,
                "tracks": [],
                "events": []
            })
            self._current_pattern = self.patterns[-1]
            self._pattern_index += 1

        # Enrich event with pattern index for convenience
        ev = dict(ev)
        ev["pattern_index"] = int(self._current_pattern["pattern_index"])
        self._current_pattern["events"].append(ev)

    def finalize(self):
        out = {
            "run_meta": self.run_meta,
            "patterns": self.patterns
        }
        with open(self.run_path, "w") as f:
            json.dump(out, f, indent=2)
        return self.run_path

# ---------------- Main -----------------
def main():
    args = parse_arguments()

    # seed RNGs deterministically (if seed is None, Python will seed from system)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    global_rng = random.Random(args.seed)

    # prepare run logging
    # One JSON file per run; patterns are logged as objects with metadata then events.
    # (RunLogger will be finalized on shutdown.)
    tmp_run_meta = {
        "seed": args.seed,
        "bpm": args.bpm,
        "steps": args.steps,
        "channels": args.channels,
        "sr": args.sr,
        "blocksize": args.blocksize,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    run_logger = RunLogger(tmp_run_meta)

    # basic checks
    if not os.path.exists(args.db_path):
        print(f"Error: Database file '{args.db_path}' not found!")
        sys.exit(1)
    if not os.path.exists(args.samples_dir):
        print(f"Error: Samples directory '{args.samples_dir}' not found!")
        sys.exit(1)

    print("Loading samples (deterministic)...")
    samples_by_type = load_samples_from_db(args.db_path, args.samples_dir, args.samples_per_track, global_rng)
    total_samples = sum(len(v) for v in samples_by_type.values())
    if total_samples == 0:
        print("Error: No samples found in database!")
        sys.exit(1)
    print(f"Loaded {total_samples} samples total.")

    # create sequencer & engine
    sequencer = Sequencer(samples_by_type, args, global_rng, run_logger)
    engine = AudioEngine(sequencer, args, global_rng)

    try:
        engine.start()
        print("Sequencer running. Press Ctrl+C to stop.")
        while True:
            try:
                time.sleep(1.0)
            except KeyboardInterrupt:
                break
    finally:
        print("Stopping...")
        engine.stop()
        # write run logs (single JSON file)
        run_path = run_logger.finalize()
        print(f"Run logs written to: {run_path}")

if __name__ == "__main__":
    main()

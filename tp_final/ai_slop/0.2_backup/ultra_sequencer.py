#!/usr/bin/env python3
"""
Deterministic Microsample Sequencer
- Clock / Sequencer / AudioEngine separation
- Per-run logging (single runs/run_YYYYMMDD_HHMMSS.json)
- Per-track stacks with memory decay
- Density drift & sample-entropy based intensity nudging
- Deterministic sample loading (no ORDER BY RANDOM)

Additions for more agentic / artistically valuable behavior:
- Director (section/form/arc control)
- Critic (self-evaluation and feedback targets)
- MotifBank (capture + variation of successful patterns)
- Sample feature extraction + lightweight clustering (palette cohesion + contrast)
- GrooveModel (swing + microtiming with stable “feel”)
- ConstraintEngine (voice limiting, continuity, phrasing)
- NoveltyEngine (recency-aware sample selection pressure)
- TensionModel (rise/release shaping)
- GestureEngine (pattern-level “moves” like stutter/freeze/reverse-tail)
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
    parser.add_argument('--density-drift-rate', type=float, default=0.002, help='Maximum density drift step per regeneration')
    parser.add_argument('--entropy-nudge', action='store_true', help='Enable sample-entropy based intensity nudging')

    # Agentic additions (kept minimal and with safe defaults)
    parser.add_argument('--enable-director', action='store_true', help='Enable section/form director')
    parser.add_argument('--enable-critic', action='store_true', help='Enable critic (self-evaluation feedback)')
    parser.add_argument('--enable-motifs', action='store_true', help='Enable motif capture + variation')
    parser.add_argument('--enable-clustering', action='store_true', help='Enable feature-based sample clustering')
    parser.add_argument('--enable-groove', action='store_true', help='Enable swing + microtiming groove model')
    parser.add_argument('--enable-constraints', action='store_true', help='Enable constraint engine (voice limiting/continuity)')
    parser.add_argument('--enable-novelty', action='store_true', help='Enable novelty engine (recency-aware selection)')
    parser.add_argument('--enable-tension', action='store_true', help='Enable tension model shaping')
    parser.add_argument('--enable-gestures', action='store_true', help='Enable gesture engine (stutter/freeze/reverse-tail)')

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
    # Agentic additions (computed after load; kept optional)
    features: Dict[str, float] = field(default_factory=dict)
    cluster_id: int = -1

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

# ---------------- Agentic: features + clustering -----------------
class SampleFeatureExtractor:
    """Compute lightweight per-sample features (deterministically)."""
    def __init__(self, sr_target: int):
        self.sr_target = sr_target

    def _to_mono(self, x: np.ndarray) -> np.ndarray:
        if x.ndim > 1:
            return np.mean(x, axis=1)
        return x

    def extract(self, sample: Sample) -> Dict[str, float]:
        x = self._to_mono(sample.data)
        if sample.sr != self.sr_target:
            x = resample_linear(x, sample.sr / self.sr_target)

        # limit analysis window for speed and determinism
        n = min(len(x), self.sr_target)  # up to 1s
        x = x[:n].astype(np.float32)
        if n <= 16:
            return {"rms": float(sample.rms), "zcr": 0.0, "centroid": 0.0, "duration": float(sample.duration)}

        # RMS
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        # ZCR (proxy for noisiness/percussiveness)
        zc = float(np.mean(np.abs(np.diff(np.signbit(x))).astype(np.float32)))

        # Spectral centroid (proxy for brightness)
        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * win)
        mag = np.abs(X).astype(np.float32) + 1e-12
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr_target).astype(np.float32)
        centroid = float(np.sum(freqs * mag) / np.sum(mag))

        return {"rms": rms, "zcr": zc, "centroid": centroid, "duration": float(sample.duration)}

class KMeansLite:
    """Tiny deterministic k-means implementation (no sklearn)."""
    def __init__(self, k: int, iters: int = 16, rng: Optional[random.Random] = None):
        self.k = max(1, int(k))
        self.iters = int(iters)
        self.rng = rng or random.Random(0)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.int32)
        k = min(self.k, n)

        # deterministic init: choose k points by stepping (no random dependence on input order beyond deterministic rng)
        idxs = list(range(n))
        self.rng.shuffle(idxs)
        centers = X[idxs[:k]].copy()

        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(self.iters):
            # assign
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1).astype(np.int32)

            if np.array_equal(new_labels, labels):
                break
            labels = new_labels

            # update
            for j in range(k):
                mask = (labels == j)
                if np.any(mask):
                    centers[j] = X[mask].mean(axis=0)
                else:
                    # re-seed empty cluster deterministically
                    centers[j] = X[self.rng.randrange(n)]
        return labels

class SampleClusterer:
    """Cluster per-stem samples into a few timbral groups to enable palette cohesion/contrast."""
    def __init__(self, rng: random.Random, k_per_stem: int = 6):
        self.rng = rng
        self.k_per_stem = k_per_stem

    def cluster(self, samples_by_type: Dict[str, List[Sample]]):
        for stem, samples in samples_by_type.items():
            if not samples:
                continue
            feats = []
            for s in samples:
                f = s.features or {}
                feats.append([f.get("rms", 0.0), f.get("zcr", 0.0), f.get("centroid", 0.0), f.get("duration", 0.0)])
            X = np.asarray(feats, dtype=np.float32)
            # normalize columns (avoid dominance)
            mu = X.mean(axis=0)
            sig = X.std(axis=0) + 1e-6
            Xn = (X - mu) / sig

            km = KMeansLite(k=self.k_per_stem, iters=20, rng=random.Random(self.rng.randint(0, 2**30)))
            labels = km.fit_predict(Xn)
            for s, cid in zip(samples, labels.tolist()):
                s.cluster_id = int(cid)

# ---------------- Agentic: tension -----------------
class TensionModel:
    """Single scalar tension (0..1) shaped by events and section targets."""
    def __init__(self):
        self.value = 0.3

    def update_from_event(self, ev: Dict[str, Any]):
        # very simple: more density/intensity/reverse pushes tension upward a bit
        inc = 0.0
        inc += 0.01 * float(ev.get("density", 0.5))
        inc += 0.01 * min(6.0, float(ev.get("intensity", 1.0))) / 6.0
        inc += 0.01 if ev.get("reversed") else 0.0
        self.value = float(np.clip(self.value + inc, 0.0, 1.0))

    def relax(self, amount: float):
        self.value = float(np.clip(self.value - abs(amount), 0.0, 1.0))

# ---------------- Agentic: director (sections/arcs) -----------------
@dataclass
class DirectorState:
    name: str
    target_density: Tuple[float, float]
    target_intensity: Tuple[float, float]
    reverse_prob_mul: float
    lowpass_range: Tuple[float, float]
    max_voices_per_step: int
    swing: float
    section_len_patterns: Tuple[int, int]  # range

class Director:
    """Form generator: chooses sections and target ranges, giving the system long-term intent."""
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.patterns_left = 0
        self.state = self._new_state("intro")

    def _new_state(self, name: str) -> DirectorState:
        # Tuned to be musically legible but still flexible.
        if name == "intro":
            return DirectorState("intro", (0.10, 0.35), (0.8, 2.0), 0.6, (1500.0, 9000.0), 2, 0.05, (2, 5))
        if name == "build":
            return DirectorState("build", (0.25, 0.60), (1.5, 4.0), 1.0, (4000.0, 16000.0), 3, 0.10, (3, 7))
        if name == "drop":
            return DirectorState("drop", (0.45, 0.85), (3.0, 6.0), 1.2, (8000.0, 20000.0), 4, 0.15, (3, 8))
        if name == "breakdown":
            return DirectorState("breakdown", (0.05, 0.30), (0.8, 2.5), 1.4, (800.0, 6000.0), 2, 0.08, (2, 6))
        # default
        return DirectorState(name, (0.20, 0.55), (1.0, 4.0), 1.0, (2000.0, 14000.0), 3, 0.10, (3, 7))

    def _choose_next_name(self, current: str, tension: float, boredom: float) -> str:
        # Simple agentic policy: if bored, seek contrast; if tension high, release; else escalate sometimes.
        if tension > 0.75:
            return "breakdown"
        if boredom > 0.65:
            return self.rng.choice(["breakdown", "build", "drop"])
        if current in ("intro", "breakdown"):
            return self.rng.choice(["build", "build", "drop"])
        if current == "build":
            return self.rng.choice(["drop", "drop", "breakdown"])
        if current == "drop":
            return self.rng.choice(["breakdown", "build"])
        return self.rng.choice(["build", "drop", "breakdown"])

    def maybe_advance(self, tension: float, boredom: float):
        if self.patterns_left > 0:
            self.patterns_left -= 1
            return
        next_name = self._choose_next_name(self.state.name, tension, boredom)
        self.state = self._new_state(next_name)
        self.patterns_left = self.rng.randint(*self.state.section_len_patterns)

# ---------------- Agentic: critic (self-evaluation) -----------------
@dataclass
class CriticReport:
    boredom: float
    repetition: float
    density: float
    syncopation: float
    notes: Dict[str, float] = field(default_factory=dict)

class Critic:
    """Analyzes recent events and produces feedback signals."""
    def __init__(self):
        pass

    def _entropy(self, items: List[Any]) -> float:
        if not items:
            return 0.0
        counts = Counter(items)
        n = float(len(items))
        ps = np.array([c / n for c in counts.values()], dtype=np.float32)
        return float(-np.sum(ps * np.log2(ps + 1e-12)))

    def analyze_pattern(self, pattern_obj: Dict[str, Any]) -> CriticReport:
        events = pattern_obj.get("events", [])
        if not events:
            return CriticReport(boredom=0.2, repetition=0.0, density=0.0, syncopation=0.0, notes={"empty": 1.0})

        names = [e.get("sample_name", "") for e in events]
        step_idxs = [int(e.get("step_index", 0)) for e in events]

        ent = self._entropy(names)
        # Repetition measure: low entropy => high repetition
        repetition = float(np.clip(1.0 - (ent / 4.0), 0.0, 1.0))

        # Density measure: events per step (normalized)
        steps = 16
        if events:
            steps = max(1, max(step_idxs) + 1)
        density = float(np.clip(len(events) / float(steps * 4), 0.0, 1.0))  # rough normalization

        # Syncopation: share of events on offbeats (odd steps in 16th grid)
        off = sum(1 for s in step_idxs if (s % 2) == 1)
        sync = float(off / max(1, len(step_idxs)))

        # Boredom heuristic: repetition + low sync + low density
        boredom = float(np.clip(0.55 * repetition + 0.25 * (1.0 - sync) + 0.20 * (1.0 - density), 0.0, 1.0))

        return CriticReport(boredom=boredom, repetition=repetition, density=density, syncopation=sync,
                            notes={"entropy": ent})

# ---------------- Agentic: motifs -----------------
@dataclass
class Motif:
    stem_type: str
    steps: List[Dict[str, Any]]
    created_at: float
    score: float

class MotifBank:
    """Stores good patterns and can produce variations."""
    def __init__(self, rng: random.Random, max_motifs_per_stem: int = 16):
        self.rng = rng
        self.max_motifs_per_stem = max_motifs_per_stem
        self.bank: Dict[str, List[Motif]] = {"drums": [], "bass": [], "other": [], "vocals": []}

    def consider_capture(self, sequencer: "Sequencer", critic_report: CriticReport):
        # Capture when boredom is low (i.e., it felt good) and it actually had content.
        if critic_report.boredom > 0.35:
            return
        # snapshot each track’s steps
        for t in sequencer.tracks:
            steps = []
            for s in t.pattern.steps:
                steps.append({
                    "index": int(s.index),
                    "sample_idx": int(s.sample_idx),
                    "on": bool(s.on),
                    "prob": float(s.prob),
                    "semitone": float(s.semitone),
                    "gain": float(s.gain),
                    "lowpass": float(s.lowpass),
                })
            m = Motif(stem_type=t.stem_type, steps=steps, created_at=time.time(), score=float(1.0 - critic_report.boredom))
            lst = self.bank[t.stem_type]
            lst.append(m)
            lst.sort(key=lambda x: x.score, reverse=True)
            if len(lst) > self.max_motifs_per_stem:
                del lst[self.max_motifs_per_stem:]

    def _vary_steps(self, steps: List[Dict[str, Any]], steps_len: int) -> List[Dict[str, Any]]:
        out = [dict(s) for s in steps]

        # rotate sometimes
        if self.rng.random() < 0.35:
            shift = self.rng.randrange(steps_len)
            out = out[-shift:] + out[:-shift]

        # mutate a few steps
        mut_n = 1 + (1 if self.rng.random() < 0.6 else 0) + (1 if self.rng.random() < 0.25 else 0)
        for _ in range(mut_n):
            i = self.rng.randrange(steps_len)
            out[i]["on"] = bool(out[i]["on"] if self.rng.random() < 0.7 else (not out[i]["on"]))
            out[i]["prob"] = float(np.clip(out[i]["prob"] + self.rng.uniform(-0.15, 0.15), 0.05, 1.0))
            out[i]["gain"] = float(np.clip(out[i]["gain"] * self.rng.uniform(0.85, 1.15), 0.01, 1.0))
            # lowpass wobble
            out[i]["lowpass"] = float(np.clip(out[i]["lowpass"] * self.rng.uniform(0.8, 1.2), 200.0, 20000.0))

        return out

    def maybe_apply(self, track: "Track", steps_len: int, apply_prob: float = 0.25):
        motifs = self.bank.get(track.stem_type, [])
        if not motifs:
            return
        if self.rng.random() > apply_prob:
            return

        # biased toward best motifs
        best = motifs[:min(6, len(motifs))]
        m = best[self.rng.randrange(len(best))]
        varied = self._vary_steps(m.steps, steps_len)

        # write back into track.pattern
        for i, s in enumerate(track.pattern.steps):
            v = varied[i % steps_len]
            s.sample_idx = int(v["sample_idx"])
            s.on = bool(v["on"])
            s.prob = float(v["prob"])
            s.semitone = float(v["semitone"])
            s.gain = float(v["gain"])
            s.lowpass = float(v["lowpass"])

# ---------------- Agentic: groove -----------------
class GrooveModel:
    """Stable feel per section: swing + microtiming. Deterministic via RNG."""
    def __init__(self, rng: random.Random, sr: int, clock_samples_per_step: int):
        self.rng = rng
        self.sr = sr
        self.samples_per_step = max(1, int(clock_samples_per_step))
        self.swing = 0.0  # 0..~0.25 (fraction of step)
        self.micro = 0.0  # 0..~0.10 (fraction of step)
        self.per_track_bias: Dict[int, float] = {}

    def set_targets(self, swing: float, micro: float):
        self.swing = float(np.clip(swing, 0.0, 0.30))
        self.micro = float(np.clip(micro, 0.0, 0.20))

    def offset_samples(self, track_id: int, step_index: int) -> int:
        if track_id not in self.per_track_bias:
            # stable bias per track across a section
            self.per_track_bias[track_id] = self.rng.uniform(-0.5, 0.5)

        # swing: delay offbeats (odd steps) by swing * step
        swing_off = (self.swing * self.samples_per_step) if (step_index % 2 == 1) else 0.0

        # micro: small +/- jitter with stable per-track bias
        jitter = (self.rng.uniform(-1.0, 1.0) * 0.5 + self.per_track_bias[track_id]) * (self.micro * self.samples_per_step)

        return int(round(swing_off + jitter))

# ---------------- Agentic: constraints -----------------
class ConstraintEngine:
    """Applies musical constraints to decisions (e.g., max voices, continuity)."""
    def __init__(self, max_voices_per_step: int = 3):
        self.max_voices_per_step = max(1, int(max_voices_per_step))

    def set_max_voices(self, n: int):
        self.max_voices_per_step = max(1, int(n))

    def prune(self, decisions: List[Tuple["Track", Sample, bool, bool, "Step"]], rng: random.Random) -> List[Tuple["Track", Sample, bool, bool, "Step"]]:
        if len(decisions) <= self.max_voices_per_step:
            return decisions
        # heuristic: keep drums/bass with higher priority
        priority = {"drums": 3, "bass": 2, "other": 1, "vocals": 1}
        scored = []
        for d in decisions:
            t = d[0]
            s = d[4]
            sc = priority.get(t.stem_type, 1) + 0.5 * float(s.prob) + rng.random() * 0.01
            scored.append((sc, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:self.max_voices_per_step]]

# ---------------- Agentic: novelty -----------------
class NoveltyEngine:
    """Recency-aware pressure to avoid overused samples (soft bias, not a hard ban)."""
    def __init__(self, rng: random.Random, memory: int = 64):
        self.rng = rng
        self.memory = max(8, int(memory))
        self.recent_by_track: Dict[int, deque] = {}

    def note_play(self, track_id: int, sample_name: str):
        if track_id not in self.recent_by_track:
            self.recent_by_track[track_id] = deque(maxlen=self.memory)
        self.recent_by_track[track_id].append(sample_name)

    def pick_index(self, track: "Track", base_index: int) -> int:
        # If no samples or no memory, keep deterministic default.
        if not track.samples:
            return base_index
        rec = self.recent_by_track.get(track.id)
        if not rec:
            return base_index

        # try a handful of candidates and choose the least-recently-used by name
        tries = min(8, len(track.samples))
        best_idx = base_index % len(track.samples)
        best_score = 1e9

        # include the base choice as candidate
        candidates = [best_idx]
        for _ in range(tries - 1):
            candidates.append(self.rng.randrange(len(track.samples)))

        # score by last occurrence position (older is better)
        rec_list = list(rec)
        for idx in candidates:
            nm = track.samples[idx].name
            if nm in rec_list:
                # distance from end (smaller => more recent => worse)
                last_pos = len(rec_list) - 1 - (len(rec_list) - 1 - rec_list[::-1].index(nm))
                # simpler: approximate "recency" by searching from end
                # (we keep deterministic behavior; performance not critical)
                for j in range(len(rec_list) - 1, -1, -1):
                    if rec_list[j] == nm:
                        last_pos = j
                        break
                recency = len(rec_list) - 1 - last_pos
            else:
                recency = 1e6
            score = -recency  # higher recency => lower score; we want oldest => highest, so invert
            # convert to minimization with negative
            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

# ---------------- Agentic: gestures -----------------
@dataclass
class Gesture:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

class GestureEngine:
    """Occasional recognizable moves (stutter/freeze/reverse-tail) to create authored gestures."""
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.current: Optional[Gesture] = None
        self.patterns_left = 0

    def maybe_new(self, section_name: str, tension: float):
        # Keep sparse: gestures should be events, not constant.
        if self.patterns_left > 0:
            self.patterns_left -= 1
            if self.patterns_left == 0:
                self.current = None
            return

        p = 0.06 + 0.10 * tension
        if self.rng.random() > p:
            return

        # Choose gesture type depending on section
        if section_name == "drop":
            g = self.rng.choice(["stutter", "reverse_tail"])
        elif section_name == "breakdown":
            g = self.rng.choice(["freeze", "reverse_tail"])
        else:
            g = self.rng.choice(["reverse_tail", "stutter"])

        if g == "stutter":
            self.current = Gesture("stutter", {"chunk_ms": self.rng.choice([30, 40, 60]), "repeats": self.rng.choice([2, 3, 4])})
        elif g == "freeze":
            self.current = Gesture("freeze", {"slice_ms": self.rng.choice([80, 120, 180])})
        else:
            self.current = Gesture("reverse_tail", {"tail_ms": self.rng.choice([60, 120, 200])})

        self.patterns_left = self.rng.randint(1, 2)

    def apply(self, sig: np.ndarray, sr: int) -> np.ndarray:
        if self.current is None:
            return sig
        g = self.current
        if len(sig) == 0:
            return sig

        if g.name == "stutter":
            chunk = int(sr * (float(g.params.get("chunk_ms", 40)) / 1000.0))
            reps = int(g.params.get("repeats", 3))
            chunk = max(8, min(chunk, len(sig)))
            head = sig[:chunk].copy()
            out = np.concatenate([head for _ in range(reps)] + [sig], axis=0)
            return out

        if g.name == "freeze":
            sl = int(sr * (float(g.params.get("slice_ms", 120)) / 1000.0))
            sl = max(8, min(sl, len(sig)))
            # repeat a slice to create a "frozen" texture
            slice_ = sig[:sl].copy()
            out = np.concatenate([slice_ for _ in range(4)] + [sig], axis=0)
            return out

        if g.name == "reverse_tail":
            tail = int(sr * (float(g.params.get("tail_ms", 120)) / 1000.0))
            tail = max(8, min(tail, len(sig)))
            head = sig[:-tail]
            tail_sig = sig[-tail:][::-1]
            return np.concatenate([head, tail_sig], axis=0)

        return sig

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

# ---------------- Run logging -----------------
class RunLogger:
    """
    Per-run logger that writes a single JSON file.
    Each run is one .json file, and each pattern is an object (metadata + events).
    """
    def __init__(self, run_meta: Dict[str, Any], runs_dir: Path = Path("runs")):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_path = self.runs_dir / f"run_{ts}.json"

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

    def start_pattern(self, sequencer: "Sequencer", reason: str, extra: Optional[Dict[str, Any]] = None):
        if self._current_pattern is not None:
            self._current_pattern["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        pat = {
            "pattern_index": int(self._pattern_index),
            "reason": str(reason),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "density": float(sequencer.density),
            "intensity": float(sequencer.intensity),
            "loop_target": int(sequencer.loop_target),
            "director": getattr(sequencer, "director_state_name", None),
            "tension": float(getattr(sequencer, "tension_value", 0.0)),
            "tracks": self._snapshot_tracks_pattern(sequencer),
            "events": []
        }
        if extra:
            pat.update(extra)

        self.patterns.append(pat)
        self._current_pattern = pat
        self._pattern_index += 1

    def log_event(self, ev: Dict[str, Any]):
        if self._current_pattern is None:
            # fallback
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

        ev = dict(ev)
        ev["pattern_index"] = int(self._current_pattern["pattern_index"])
        self._current_pattern["events"].append(ev)

    def finalize(self):
        out = {"run_meta": self.run_meta, "patterns": self.patterns}
        with open(self.run_path, "w") as f:
            json.dump(out, f, indent=2)
        return self.run_path

# ---------------- Sequencer -----------------
class Sequencer:
    def __init__(self, samples_by_type: Dict[str, List[Sample]], config: argparse.Namespace, rng: random.Random, run_logger: RunLogger):
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

        # Agentic systems (enabled by flags)
        self.tension = TensionModel() if config.enable_tension else None
        self.director = Director(self.rng) if config.enable_director else None
        self.critic = Critic() if config.enable_critic else None
        self.motifs = MotifBank(self.rng) if config.enable_motifs else None
        self.constraints = ConstraintEngine() if config.enable_constraints else None
        self.novelty = NoveltyEngine(self.rng) if config.enable_novelty else None
        self.gestures = GestureEngine(self.rng) if config.enable_gestures else None

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

        # Expose some values for logger snapshot
        self.director_state_name = self.director.state.name if self.director else None
        self.tension_value = self.tension.value if self.tension else 0.0

        # Start the first pattern block in the run log (metadata FIRST, then events)
        self.run_logger.start_pattern(self, reason="init", extra={"agentic": self._agentic_flags_snapshot()})

    def _agentic_flags_snapshot(self) -> Dict[str, Any]:
        return {
            "director": bool(self.director),
            "critic": bool(self.critic),
            "motifs": bool(self.motifs),
            "constraints": bool(self.constraints),
            "novelty": bool(self.novelty),
            "tension": bool(self.tension),
            "gestures": bool(self.gestures),
        }

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

                # novelty engine can bias selection away from recent repeats
                if self.novelty:
                    samp_idx = self.novelty.pick_index(track, samp_idx)

                chosen_sample = track.samples[samp_idx]

            # chance to reverse (director can modulate)
            reverse_prob = self.reverse_prob
            if self.director:
                reverse_prob = max(0.0, min(1.0, reverse_prob * float(self.director.state.reverse_prob_mul)))
            reversed_play = (track.rng.random() < reverse_prob)

            results.append((track, chosen_sample, use_stack_choice, reversed_play, step))

        # constraints can prune over-dense steps
        if self.constraints:
            results = self.constraints.prune(results, self.rng)

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
            "sample_cluster": int(getattr(sample, "cluster_id", -1)),
            "gain": float(step_obj.gain),
            "semitone": float(step_obj.semitone),
            "lowpass": float(step_obj.lowpass),
            "prob": float(step_obj.prob),
            "used_stack": bool(used_stack),
            "reversed": bool(reversed_play),
            "density": float(self.density),
            "intensity": float(self.intensity),
            "director": self.director.state.name if self.director else None,
            "tension": float(self.tension.value) if self.tension else None,
            "time": time.time()
        }
        track.events.append(ev)

        # novelty memory learns what we actually played
        if self.novelty:
            self.novelty.note_play(track.id, sample.name)

        # tension model updates from events
        if self.tension:
            self.tension.update_from_event(ev)
            self.tension_value = self.tension.value

        # Log under the currently active pattern object
        self.run_logger.log_event(ev)

    def _apply_director_targets_to_tracks(self):
        if not self.director:
            return
        st = self.director.state
        # per-track: constrain lowpass within section aesthetic
        for t in self.tracks:
            for s in t.pattern.steps:
                # only meaningful if it triggers
                s.lowpass = float(self.rng.uniform(*st.lowpass_range))

        # constraints: max voices
        if self.constraints:
            self.constraints.set_max_voices(st.max_voices_per_step)

    def _apply_entropy_nudge(self):
        # keep your original logic, but make it explicitly a post-analysis “nudge”
        if not self.config.entropy_nudge:
            return
        for track in self.tracks:
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

    def regenerate_patterns(self):
        """Regenerate patterns for all tracks; called when loop finishes."""

        # --- Critic feedback on the pattern that just ended ---
        boredom = 0.0
        critic_report = None
        if self.critic and self.run_logger.patterns:
            critic_report = self.critic.analyze_pattern(self.run_logger.patterns[-1])
            boredom = critic_report.boredom

        # --- Director: section transitions based on tension + boredom ---
        if self.director:
            tension_val = self.tension.value if self.tension else 0.0
            self.director.maybe_advance(tension=tension_val, boredom=boredom)
            self.director_state_name = self.director.state.name

        # optionally drift density
        if self.config.enable_density_drift:
            delta = self.rng.uniform(-self.config.density_drift_rate, self.config.density_drift_rate)
            self.density = min(max(0.01, self.density + delta), 0.99)
        else:
            self.density = self.density  # unchanged

        # intensity variation
        self.intensity = self.rng.uniform(1.0, 5.0)

        # director targets override global density/intensity into section ranges
        if self.director:
            d0, d1 = self.director.state.target_density
            i0, i1 = self.director.state.target_intensity
            # blend current with target ranges to keep continuity
            self.density = float(np.clip(0.5 * self.density + 0.5 * self.rng.uniform(d0, d1), 0.01, 0.99))
            self.intensity = float(np.clip(0.5 * self.intensity + 0.5 * self.rng.uniform(i0, i1), 0.5, 6.0))

        # enforce entropy-driven nudging (original behavior, now in a helper)
        self._apply_entropy_nudge()

        for track in self.tracks:
            track.pattern = Pattern.random(
                self.steps,
                max(1, len(track.samples)),
                rng=track.rng,
                pattern_density=self.density,
                pattern_intensity=self.intensity
            )

        # motifs: sometimes reuse/transform a stored good idea
        if self.motifs:
            # capture motifs from the pattern that just ended if it was good
            if critic_report is not None:
                self.motifs.consider_capture(self, critic_report)
            # apply motifs as variation into new patterns sometimes
            for track in self.tracks:
                self.motifs.maybe_apply(track, steps_len=self.steps, apply_prob=0.22)

        # director lowpass etc.
        self._apply_director_targets_to_tracks()

        # tension: if we moved into breakdown, relax
        if self.tension and self.director and self.director.state.name == "breakdown":
            self.tension.relax(0.15)
            self.tension_value = self.tension.value

        # gestures: maybe create a new “move” for the next pattern(s)
        if self.gestures and self.director:
            self.gestures.maybe_new(section_name=self.director.state.name, tension=(self.tension.value if self.tension else 0.0))

        # choose new loop length
        self.loop_target = self.rng.randint(self.config.repeat_min, self.config.repeat_max)
        print(f"[Regenerated patterns. section={self.director_state_name}, density={self.density:.3f}, intensity={self.intensity:.3f}, next_repeat={self.loop_target}]")

        # logging: start a new pattern object and snapshot its metadata BEFORE any events occur
        extra = {}
        if critic_report is not None:
            extra["critic"] = {
                "boredom": critic_report.boredom,
                "repetition": critic_report.repetition,
                "density": critic_report.density,
                "syncopation": critic_report.syncopation,
                "notes": critic_report.notes,
            }
        if self.gestures and self.gestures.current is not None:
            extra["gesture"] = {"name": self.gestures.current.name, "params": self.gestures.current.params}
        self.run_logger.start_pattern(self, reason="regenerate", extra=extra)

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

        # groove model (needs samples_per_step)
        self.groove = None
        if self.config.enable_groove:
            self.groove = GrooveModel(self.rng, self.sr, self.clock.samples_per_step)
            # initialize with safe default; director can influence later through state swing
            self.groove.set_targets(swing=0.10, micro=0.03)

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

    def schedule_play(self, sample: Sample, step_sample_pos: int, channel: int, reversed_play: bool, step: Step, offset_samples: int = 0):
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

        # gestures operate on the audio chunk level
        if self.seq.gestures:
            sig = self.seq.gestures.apply(sig, self.sr)

        sig = apply_adsr_envelope(sig, self.sr, self.seq.env_attack, self.seq.env_decay, self.seq.env_sustain, self.seq.env_release)
        sig = apply_fade(sig, self.fade_samples, self.sr)

        # microtiming: implement as leading silence padding
        if offset_samples != 0:
            # negative offset means "early": we cannot time-travel in this simple scheduler,
            # so we clamp early hits to no delay (still deterministic).
            if offset_samples > 0:
                pad = np.zeros((offset_samples,), dtype=np.float32)
                sig = np.concatenate([pad, sig], axis=0)

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

            # director influences groove targets (stable “feel”)
            if self.groove and self.seq.director:
                st = self.seq.director.state
                self.groove.set_targets(swing=st.swing, micro=0.04)

            # let sequencer decide events for this step
            decisions = self.seq.decide_for_step(step_index)

            for (track, sample, used_stack, reversed_play, step_obj) in decisions:
                # record event in sequencer logs
                self.seq.record_event(pos, step_index, track, sample, used_stack, reversed_play, step_obj)

                # groove microtiming offset (per track + step)
                offs = 0
                if self.groove:
                    offs = self.groove.offset_samples(track.id, int(step_index))

                # audio schedule
                self.schedule_play(sample, pos, track.channel, reversed_play, step_obj, offset_samples=offs)

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

# ---------------- Main -----------------
def main():
    args = parse_arguments()

    # seed RNGs deterministically (if seed is None, Python will seed from system)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    global_rng = random.Random(args.seed)

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

    # Agentic preprocessing: features + clustering (optional)
    feat = SampleFeatureExtractor(sr_target=args.sr)
    for stem, lst in samples_by_type.items():
        for s in lst:
            try:
                s.features = feat.extract(s)
            except Exception:
                s.features = {}

    if args.enable_clustering:
        clusterer = SampleClusterer(global_rng, k_per_stem=6)
        clusterer.cluster(samples_by_type)

    # prepare run logging (single .json)
    run_meta = {
        "seed": args.seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bpm": args.bpm,
        "steps": args.steps,
        "channels": args.channels,
        "sr": args.sr,
        "blocksize": args.blocksize,
        "flags": {
            "enable_density_drift": bool(args.enable_density_drift),
            "entropy_nudge": bool(args.entropy_nudge),
            "director": bool(args.enable_director),
            "critic": bool(args.enable_critic),
            "motifs": bool(args.enable_motifs),
            "clustering": bool(args.enable_clustering),
            "groove": bool(args.enable_groove),
            "constraints": bool(args.enable_constraints),
            "novelty": bool(args.enable_novelty),
            "tension": bool(args.enable_tension),
            "gestures": bool(args.enable_gestures),
        }
    }
    run_logger = RunLogger(run_meta)

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
        run_path = run_logger.finalize()
        print(f"Run logs written to: {run_path}")

if __name__ == "__main__":
    main()
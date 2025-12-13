#!/usr/bin/env python3
"""
Groove-First Agentic Microsample Music System (full rewrite, improved)

Fixes implemented (minimal hard-coding, mostly control laws / constraints):
1) Global density governor (TokenBucket):
   - caps sustained onset rate and allows short bursts
   - fill rate adapts from tempo, state.density, coherence, critic.chaos

2) Material repetition via "anchors" (per-segment return pool):
   - each segment picks a small anchor set per stem (from current palette)
   - sample selection probabilistically returns to anchors based on commitment*coherence

3) Anti-chaos scheduling constraints:
   - global near-simultaneous onset cap
   - per-stem cooldown derived from tempo & coherence (not fixed ms)
   - token costs per stem (prevents all stems dense forever)

4) More stable groove generation:
   - per-stem Euclidean-ish motif on 8/16 subdivisions
   - controlled fill/skip probabilities reduced when coherent
   - motif mutation on bar boundaries (tracked deterministically)

Still: event-based (not a step sequencer). Onsets are chosen from soft grid candidates
with microtiming, but are not forced to land on every step.

Dependencies: numpy, soundfile, sounddevice (tkinter optional)
DB schema assumed: samples(sample_filename, stem_type, sample_position, duration, metadata_json)
"""

import os
import sys
import math
import json
import time
import heapq
import sqlite3
import random
import threading
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque, Counter

try:
    import tkinter as tk
    from tkinter import ttk
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

import numpy as np
import soundfile as sf
import sounddevice as sd


# -------------------- CLI (minimal) --------------------
def parse_arguments():
    p = argparse.ArgumentParser(description="Groove-First Agentic Microsample Music System")

    # I/O
    p.add_argument("--db-path", default="microsamples.db", help="SQLite database path")
    p.add_argument("--samples-dir", default="samples", help="Directory containing sample files")
    p.add_argument("--samples-limit", type=int, default=800, help="Max samples per stem type loaded from DB")

    # Audio
    p.add_argument("--channels", type=int, choices=[1, 2, 4], default=2, help="Output channels: 1/2/4")
    p.add_argument("--sr", type=int, default=44100, help="Sample rate")
    p.add_argument("--blocksize", type=int, default=1024, help="Audio block size")
    p.add_argument("--latency", default="high", help="sounddevice latency setting")

    # Determinism
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")

    return p.parse_args()


# -------------------- Channel mapping --------------------
def channel_map_for(output_channels: int) -> Dict[str, int]:
    if output_channels == 1:
        return {"drums": 0, "bass": 0, "other": 0, "vocals": 0}
    if output_channels == 2:
        return {"drums": 0, "bass": 0, "other": 1, "vocals": 1}
    return {"drums": 0, "bass": 1, "other": 2, "vocals": 3}


# -------------------- DSP helpers --------------------
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

    out = np.zeros((new_n, signal.shape[1]), dtype=signal.dtype)
    for c in range(signal.shape[1]):
        out[:, c] = (1 - frac) * signal[i0, c] + frac * signal[i1, c]
    return out


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        return np.mean(x, axis=1)
    return x


def apply_fade(signal: np.ndarray, fade_samples: int) -> np.ndarray:
    if signal.shape[0] <= 1 or fade_samples <= 0:
        return signal
    n = signal.shape[0]
    f = min(fade_samples, n // 2)
    if f <= 0:
        return signal
    fade_in = np.linspace(0.0, 1.0, f, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, f, dtype=np.float32)
    signal[:f] *= fade_in
    signal[-f:] *= fade_out
    return signal


def lowpass_1pole(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    x = to_mono(x).astype(np.float32, copy=False)
    cutoff = float(np.clip(cutoff_hz, 30.0, sr * 0.45))
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * math.pi * cutoff)
    a = dt / (rc + dt)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = acc + a * (x[i] - acc)
        y[i] = acc
    return y


def highpass_1pole(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    x = to_mono(x).astype(np.float32, copy=False)
    cutoff = float(np.clip(cutoff_hz, 10.0, sr * 0.45))
    dt = 1.0 / sr
    rc = 1.0 / (2.0 * math.pi * cutoff)
    a = rc / (rc + dt)
    y = np.empty_like(x)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(len(x)):
        y0 = a * (prev_y + x[i] - prev_x)
        y[i] = y0
        prev_x = x[i]
        prev_y = y0
    return y


def soft_clip(x: np.ndarray, drive: float) -> np.ndarray:
    return np.tanh(x * float(drive)).astype(np.float32, copy=False)


def envelope_perc(length: int, sr: int, attack_s: float, release_s: float) -> np.ndarray:
    a = max(1, int(round(attack_s * sr)))
    r = max(1, int(round(release_s * sr)))
    n = int(length)
    env = np.ones(n, dtype=np.float32)
    a = min(a, n)
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    r = min(r, n)
    env[-r:] *= np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


# -------------------- Rhythm helpers --------------------
def seconds_per_beat(bpm: float) -> float:
    return 60.0 / max(1e-6, float(bpm))


def euclidean_rhythm(k: int, n: int) -> List[int]:
    n = max(1, int(n))
    k = int(np.clip(k, 0, n))
    if k == 0:
        return [0] * n
    if k == n:
        return [1] * n
    pattern = []
    bucket = 0
    for _ in range(n):
        bucket += k
        if bucket >= n:
            bucket -= n
            pattern.append(1)
        else:
            pattern.append(0)
    return pattern


def rotate_list(x: List[int], r: int) -> List[int]:
    if not x:
        return x
    r = int(r) % len(x)
    return x[-r:] + x[:-r]


def quantized_time_candidates(
    now_sec: float,
    bpm: float,
    subdiv: int,
    swing: float,
    horizon_beats: float = 2.0,
) -> List[float]:
    """
    Candidate onset times on a soft grid of subdiv steps per beat, with swing on off-steps.
    """
    spb = seconds_per_beat(bpm)
    subdiv = max(1, int(subdiv))
    step_beats = 1.0 / float(subdiv)
    horizon_steps = int(math.ceil(horizon_beats / step_beats))

    beat_now = now_sec / spb
    next_step = math.floor(beat_now / step_beats) * step_beats + step_beats

    out: List[float] = []
    for i in range(horizon_steps):
        b = next_step + i * step_beats
        step_index = int(round(b / step_beats))
        is_swung = (step_index % 2 == 1)
        swing_beats = (0.0 if not is_swung else (float(swing) * step_beats * 0.9))
        t = (b + swing_beats) * spb
        if t > now_sec:
            out.append(float(t))
    return out


# -------------------- Token bucket governor --------------------
class TokenBucket:
    def __init__(self, capacity: float, fill_rate: float):
        self.capacity = float(max(1e-6, capacity))
        self.tokens = float(self.capacity)
        self.fill_rate = float(max(0.0, fill_rate))  # tokens per second
        self.last_t = 0.0

    def set_rate(self, fill_rate: float, capacity: Optional[float] = None):
        self.fill_rate = float(max(0.0, fill_rate))
        if capacity is not None:
            self.capacity = float(max(1e-6, capacity))
            self.tokens = min(self.tokens, self.capacity)

    def tick(self, now_sec: float):
        dt = max(0.0, float(now_sec) - float(self.last_t))
        self.last_t = float(now_sec)
        self.tokens = min(self.capacity, self.tokens + self.fill_rate * dt)

    def consume(self, amount: float) -> bool:
        amount = float(max(0.0, amount))
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


# -------------------- Data model --------------------
@dataclass
class Sample:
    data: np.ndarray
    sr: int
    filename: str
    name: str
    stem_type: str
    position: float
    duration: float
    meta: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)
    cluster_id: int = -1


@dataclass
class RenderParams:
    gain: float
    semitone: float
    lowpass_hz: float
    highpass_hz: float
    reverse: bool
    drive: float
    fade_s: float
    env_attack_s: float
    env_release_s: float


@dataclass
class ScheduledAudio:
    start_sample: int
    channel: int
    buffer: np.ndarray
    event_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PlayEvent:
    buffer: np.ndarray
    channel: int
    pos: int = 0
    delay: int = 0


# -------------------- Sample DB --------------------
class SampleDatabase:
    def __init__(self, db_path: str, samples_dir: str, rng: random.Random):
        self.db_path = db_path
        self.samples_dir = samples_dir
        self.rng = rng
        self.conn = sqlite3.connect(db_path)

    def load_by_stem(self, stem_type: str, limit: int) -> List[Sample]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT sample_filename, stem_type, sample_position, duration, metadata_json
            FROM samples
            WHERE stem_type = ?
            ORDER BY sample_filename ASC
            """,
            (stem_type,),
        )
        rows = cur.fetchall()
        self.rng.shuffle(rows)
        rows = rows[: max(0, int(limit))]

        out: List[Sample] = []
        for (fn, st, pos, dur, meta_json) in rows:
            path = os.path.join(self.samples_dir, fn)
            if not os.path.exists(path):
                continue
            try:
                data, sr = sf.read(path, always_2d=False)
                data = np.asarray(data, dtype=np.float32)
                meta = json.loads(meta_json) if meta_json else {}
                name = f"{meta.get('artist', 'Unknown')} - {meta.get('song_title', 'Unknown')} - {st}"
                duration = float(dur) if dur is not None else float(len(data) / max(1, sr))
                out.append(
                    Sample(
                        data=data,
                        sr=int(sr),
                        filename=str(fn),
                        name=name,
                        stem_type=str(st),
                        position=float(pos) if pos is not None else 0.0,
                        duration=duration,
                        meta=meta,
                    )
                )
            except Exception:
                continue
        return out

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# -------------------- Features + clustering --------------------
class FeatureExtractor:
    def __init__(self, sr_target: int):
        self.sr_target = int(sr_target)

    def extract(self, s: Sample) -> Dict[str, float]:
        x = to_mono(s.data)
        if s.sr != self.sr_target:
            x = resample_linear(x, s.sr / self.sr_target)
        n = min(len(x), self.sr_target)
        x = x[:n].astype(np.float32, copy=False)
        if n < 64:
            return {"rms": 0.0, "zcr": 0.0, "centroid": 0.0, "dur": float(s.duration)}

        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(x))).astype(np.float32)))

        win = np.hanning(n).astype(np.float32)
        X = np.fft.rfft(x * win)
        mag = np.abs(X).astype(np.float32) + 1e-12
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr_target).astype(np.float32)
        centroid = float(np.sum(freqs * mag) / np.sum(mag))

        return {"rms": rms, "zcr": zcr, "centroid": centroid, "dur": float(s.duration)}


class KMeansLite:
    def __init__(self, k: int, iters: int, rng: random.Random):
        self.k = max(1, int(k))
        self.iters = max(1, int(iters))
        self.rng = rng

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if n == 0:
            return np.zeros((0,), dtype=np.int32)
        k = min(self.k, n)
        idx = list(range(n))
        self.rng.shuffle(idx)
        centers = X[idx[:k]].copy()

        labels = np.zeros((n,), dtype=np.int32)
        for _ in range(self.iters):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1).astype(np.int32)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if np.any(mask):
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[self.rng.randrange(n)]
        return labels


class Clusterer:
    def __init__(self, rng: random.Random):
        self.rng = rng

    @staticmethod
    def _dynamic_k(n: int) -> int:
        if n <= 8:
            return 1
        return int(np.clip(round(math.sqrt(n)), 2, 10))

    def assign(self, samples_by_stem: Dict[str, List[Sample]]):
        for _, samples in samples_by_stem.items():
            if not samples:
                continue
            feats = []
            for s in samples:
                f = s.features
                feats.append([f.get("rms", 0.0), f.get("zcr", 0.0), f.get("centroid", 0.0), f.get("dur", 0.0)])
            X = np.asarray(feats, dtype=np.float32)
            mu = X.mean(axis=0)
            sig = X.std(axis=0) + 1e-6
            Xn = (X - mu) / sig

            k = self._dynamic_k(len(samples))
            km = KMeansLite(k=k, iters=24, rng=random.Random(self.rng.randint(0, 2**30)))
            labels = km.fit_predict(Xn)
            for s, cid in zip(samples, labels.tolist()):
                s.cluster_id = int(cid)


class DatasetProfile:
    def __init__(self, samples_by_stem: Dict[str, List[Sample]]):
        self.samples_by_stem = samples_by_stem
        self.stats: Dict[str, Dict[str, float]] = {}
        self._build()

    @staticmethod
    def _q(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        return float(np.quantile(np.asarray(values, dtype=np.float32), q))

    def _build_for(self, samples: List[Sample]) -> Dict[str, float]:
        cent = [float(s.features.get("centroid", 0.0)) for s in samples]
        dur = [float(s.features.get("dur", s.duration)) for s in samples]
        rms = [float(s.features.get("rms", 0.0)) for s in samples]
        zcr = [float(s.features.get("zcr", 0.0)) for s in samples]
        return {
            "centroid_q20": self._q(cent, 0.20),
            "centroid_q50": self._q(cent, 0.50),
            "centroid_q80": self._q(cent, 0.80),
            "dur_q20": self._q(dur, 0.20),
            "dur_q50": self._q(dur, 0.50),
            "dur_q80": self._q(dur, 0.80),
            "rms_q50": self._q(rms, 0.50),
            "zcr_q50": self._q(zcr, 0.50),
            "count": float(len(samples)),
        }

    def _build(self):
        all_samples: List[Sample] = []
        for lst in self.samples_by_stem.values():
            all_samples.extend(lst)
        self.stats["_all"] = self._build_for(all_samples)


# -------------------- Run logging --------------------
class RunLogger:
    def __init__(self, run_meta: Dict[str, Any], runs_dir: Path = Path("runs")):
        runs_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = runs_dir / f"run_{ts}.json"
        self.run_meta = dict(run_meta)
        self.segments: List[Dict[str, Any]] = []
        self._cur: Optional[Dict[str, Any]] = None
        self._seg_index = 0

    def start_segment(self, segment_meta: Dict[str, Any]):
        if self._cur is not None:
            self._cur["segment_meta"]["ended_at"] = time.time()
        seg = {
            "segment_meta": {"segment_index": int(self._seg_index), "created_at": time.time(), **segment_meta},
            "events": [],
        }
        self.segments.append(seg)
        self._cur = seg
        self._seg_index += 1

    def log_event(self, event: Dict[str, Any]):
        if self._cur is None:
            self.start_segment({"reason": "implicit_init"})
        self._cur["events"].append(event)

    def finalize(self):
        if self._cur is not None:
            self._cur["segment_meta"]["ended_at"] = time.time()
        out = {"run_meta": self.run_meta, "segments": self.segments}
        with open(self.path, "w") as f:
            json.dump(out, f, indent=2)
        return self.path


# -------------------- Agentic core --------------------
@dataclass
class GlobalState:
    tempo_bpm: float
    energy: float
    tension: float
    density: float
    brightness: float
    novelty: float
    coherence: float
    commitment: float
    swing: float  # 0..0.65

    def clamp(self):
        self.tempo_bpm = float(np.clip(self.tempo_bpm, 55.0, 175.0))
        self.energy = float(np.clip(self.energy, 0.0, 1.0))
        self.tension = float(np.clip(self.tension, 0.0, 1.0))
        self.density = float(np.clip(self.density, 0.0, 1.0))
        self.brightness = float(np.clip(self.brightness, 0.0, 1.0))
        self.novelty = float(np.clip(self.novelty, 0.0, 1.0))
        self.coherence = float(np.clip(self.coherence, 0.0, 1.0))
        self.commitment = float(np.clip(self.commitment, 0.0, 1.0))
        self.swing = float(np.clip(self.swing, 0.0, 0.65))


class Memory:
    def __init__(self, per_voice: int = 160):
        self.per_voice = max(32, int(per_voice))
        self.recent_samples: Dict[str, deque] = {}
        self.recent_clusters: Dict[str, deque] = {}
        self.recent_onsets: deque = deque(maxlen=512)

    def note(self, stem: str, sample_name: str, cluster_id: int, t_sec: float):
        if stem not in self.recent_samples:
            self.recent_samples[stem] = deque(maxlen=self.per_voice)
            self.recent_clusters[stem] = deque(maxlen=self.per_voice)
        self.recent_samples[stem].append(sample_name)
        self.recent_clusters[stem].append(int(cluster_id))
        self.recent_onsets.append(float(t_sec))

    def recency_penalty(self, stem: str, sample_name: str) -> float:
        dq = self.recent_samples.get(stem)
        if not dq:
            return 0.0
        found = []
        for i in range(len(dq) - 1, -1, -1):
            if dq[i] == sample_name:
                found.append(len(dq) - 1 - i)
                if len(found) >= 8:
                    break
        if not found:
            return 0.0
        m = min(found)
        return float(np.clip(1.0 - (m / 24.0), 0.0, 1.0))

    def local_polyphony(self, t_sec: float, window: float = 0.045) -> int:
        if not self.recent_onsets:
            return 0
        c = 0
        lo = t_sec - window
        for x in reversed(self.recent_onsets):
            if x < lo:
                break
            c += 1
        return c


class Critic:
    """
    Mostly style-agnostic:
    - repetition: low entropy on sample names
    - chaos: too-high onset rate OR too many near-simultaneous onsets
    - boredom: repetition high OR onset rate too low
    """
    def __init__(self, window_events: int = 260):
        self.window_events = max(80, int(window_events))
        self._alpha = 0.05
        self._rate_mu = 6.0
        self._rate_var = 4.0

    @staticmethod
    def _entropy(items: List[Any]) -> float:
        if not items:
            return 0.0
        c = Counter(items)
        n = float(len(items))
        p = np.array([v / n for v in c.values()], dtype=np.float32)
        return float(-np.sum(p * np.log2(p + 1e-12)))

    def _update_running(self, mu: float, var: float, x: float) -> Tuple[float, float]:
        a = self._alpha
        d = x - mu
        mu2 = mu + a * d
        var2 = (1.0 - a) * var + a * (d * d)
        return float(mu2), float(max(1e-6, var2))

    def analyze(self, recent_events: List[Dict[str, Any]]) -> Dict[str, float]:
        ev = recent_events[-self.window_events :]
        if not ev:
            return {"boredom": 0.25, "chaos": 0.0, "repetition": 0.0, "event_rate": 0.0, "poly_density": 0.0}

        names = [e.get("sample_name", "") for e in ev]
        times = [float(e.get("t_sec", 0.0)) for e in ev]

        ent = self._entropy(names)

        t0, t1 = min(times), max(times)
        dt = max(1e-6, t1 - t0)
        event_rate = float(len(ev) / dt)

        self._rate_mu, self._rate_var = self._update_running(self._rate_mu, self._rate_var, event_rate)
        rate_sigma = math.sqrt(self._rate_var)

        rate_hi = self._rate_mu + 1.0 * rate_sigma
        rate_lo = max(0.0, self._rate_mu - 1.0 * rate_sigma)
        rate_band = max(0.75, rate_hi - rate_lo)
        rate_too_high = float(np.clip((event_rate - rate_hi) / rate_band, 0.0, 1.0))
        rate_too_low = float(np.clip((rate_lo - event_rate) / rate_band, 0.0, 1.0))

        # entropy target is intentionally loose; only penalize very low entropy
        repetition = float(np.clip((3.4 - ent) / 3.4, 0.0, 1.0))

        times_sorted = sorted(times)
        close = 0
        for i in range(1, len(times_sorted)):
            if (times_sorted[i] - times_sorted[i - 1]) < 0.018:
                close += 1
        poly_density = float(close / max(1, len(times_sorted) - 1))

        chaos = float(np.clip(0.55 * rate_too_high + 0.70 * poly_density, 0.0, 1.0))
        boredom = float(np.clip(0.55 * repetition + 0.45 * rate_too_low, 0.0, 1.0))

        return {
            "boredom": boredom,
            "chaos": chaos,
            "repetition": repetition,
            "event_rate": event_rate,
            "poly_density": poly_density,
            "ent_names": ent,
            "rate_mu": self._rate_mu,
            "rate_sigma": rate_sigma,
        }


class PaletteBag:
    """
    Rotating bag of clusters per stem. Higher commitment => smaller bag (more repetition).
    """
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.bag: Dict[str, List[int]] = {}

    def refresh(self, stem: str, samples: List[Sample], commitment: float):
        clusters = sorted({int(s.cluster_id) for s in samples if s.cluster_id >= 0})
        if not clusters:
            self.bag[stem] = []
            return
        max_k = max(2, int(np.clip(round(math.sqrt(len(clusters)) * 1.2), 2, 14)))
        k = int(np.clip(round((1.0 - commitment) * (max_k - 2) + 2), 2, max_k))
        cl = clusters[:]
        self.rng.shuffle(cl)
        self.bag[stem] = cl[:k]

    def allowed_clusters(self, stem: str) -> List[int]:
        return list(self.bag.get(stem, []))


class Director:
    """
    Sets continuous targets AND per-stem rhythmic intent.
    Intent is derived from targets (not named styles).
    """
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.regime_id = 0
        self.segment_ends_at = 0.0
        self._targets: Dict[str, float] = {}
        self._rhythm: Dict[str, Dict[str, Any]] = {}

    def maybe_transition(self, now_sec: float, state: GlobalState, critic: Dict[str, float]):
        if now_sec >= self.segment_ends_at or critic.get("chaos", 0.0) > 0.92 or critic.get("boredom", 0.0) > 0.92:
            self._transition(now_sec, state, critic)

    def _transition(self, now_sec: float, state: GlobalState, critic: Dict[str, float]):
        self.regime_id += 1
        chaos = float(critic.get("chaos", 0.0))
        boredom = float(critic.get("boredom", 0.0))
        repetition = float(critic.get("repetition", 0.0))

        base = 10.0 + 32.0 * state.coherence
        base *= (1.0 - 0.60 * chaos)
        base *= (1.0 - 0.25 * boredom)
        dur = float(np.clip(self.rng.uniform(0.8 * base, 1.2 * base), 6.0, 55.0))
        self.segment_ends_at = now_sec + dur

        def nudge(x, spread):
            return float(np.clip(x + self.rng.uniform(-spread, spread), 0.0, 1.0))

        t = {}
        tempo_drift = 0.02 + 0.06 * state.novelty
        t["tempo_bpm"] = float(np.clip(state.tempo_bpm * (1.0 + self.rng.uniform(-tempo_drift, tempo_drift)), 55.0, 175.0))

        t["energy"] = nudge(state.energy, 0.14)
        t["tension"] = nudge(state.tension, 0.12)

        # important: keep density naturally moderate; allow it to rise only when coherent
        dens = nudge(state.density, 0.12)
        dens *= (0.85 + 0.30 * state.coherence)
        dens *= (1.0 - 0.55 * chaos)
        t["density"] = float(np.clip(dens, 0.08, 0.85))

        t["brightness"] = nudge(state.brightness, 0.16)
        t["novelty"] = nudge(state.novelty, 0.14)
        t["coherence"] = nudge(state.coherence, 0.14)
        t["commitment"] = nudge(state.commitment, 0.16)
        t["swing"] = float(np.clip(nudge(state.swing, 0.18), 0.0, 0.65))

        # critic feedback
        if repetition > 0.25:
            t["novelty"] = float(np.clip(t["novelty"] + 0.22 * repetition, 0.0, 1.0))
            t["commitment"] = float(np.clip(t["commitment"] - 0.18 * repetition, 0.0, 1.0))
        if boredom > 0.25:
            t["brightness"] = float(np.clip(t["brightness"] + 0.18 * boredom, 0.0, 1.0))
            t["swing"] = float(np.clip(t["swing"] + 0.12 * boredom, 0.0, 0.65))
        if chaos > 0.25:
            t["coherence"] = float(np.clip(t["coherence"] + 0.30 * chaos, 0.0, 1.0))
            t["density"] = float(np.clip(t["density"] - 0.22 * chaos, 0.0, 1.0))
            t["tension"] = float(np.clip(t["tension"] - 0.12 * chaos, 0.0, 1.0))

        self._targets = t

        # rhythmic intent (derive hits from density but keep small for very short samples)
        def stem_intent(stem: str):
            energy = t["energy"]
            density = t["density"]
            coherence = t["coherence"]
            novelty = t["novelty"]

            if stem == "drums":
                subdiv = 16 if self.rng.random() < (0.50 + 0.25 * energy) else 8
                hits = int(np.clip(round(3 + 8 * density + 2 * energy), 2, 11))
                mutate_bars = int(np.clip(round(2 + 6 * coherence), 2, 8))
            elif stem == "bass":
                subdiv = 8 if self.rng.random() < 0.75 else 16
                hits = int(np.clip(round(1 + 5 * density), 1, 7))
                mutate_bars = int(np.clip(round(3 + 7 * coherence), 3, 10))
            elif stem == "vocals":
                subdiv = 8
                hits = int(np.clip(round(0 + 4 * density * (0.5 + 0.5 * novelty)), 0, 5))
                mutate_bars = int(np.clip(round(4 + 8 * coherence), 4, 12))
            else:
                subdiv = 16 if self.rng.random() < 0.55 else 8
                hits = int(np.clip(round(1 + 6 * density), 1, 9))
                mutate_bars = int(np.clip(round(3 + 7 * coherence), 3, 10))

            rotate = self.rng.randrange(max(1, subdiv))
            return {"subdiv": subdiv, "hits": hits, "mutate_bars": mutate_bars, "rotate": rotate}

        self._rhythm = {st: stem_intent(st) for st in ["drums", "bass", "other", "vocals"]}

    def current_targets(self) -> Dict[str, float]:
        return dict(self._targets)

    def rhythm_intent(self, stem: str) -> Dict[str, Any]:
        return dict(self._rhythm.get(stem, {}))

    @property
    def name(self) -> str:
        return f"regime_{self.regime_id}"


# -------------------- Voice --------------------
class Voice:
    def __init__(self, stem: str, channel: int, samples: List[Sample], rng: random.Random, memory: Memory):
        self.stem = stem
        self.channel = int(channel)
        self.samples = samples
        self.rng = rng
        self.memory = memory

        self.next_time_sec: float = 0.0

        self._motif_steps: List[int] = []
        self._motif_subdiv: int = 16
        self._motif_bars_left: int = 0
        self._motif_rotate: int = 0
        self._last_bar_index: Optional[int] = None

        self.bias_brightness = self.rng.random()
        self.bias_activity = self.rng.random()

    def _role(self) -> Dict[str, float]:
        if self.stem == "drums":
            return {"brightness": 0.60, "pitch": 0.05, "hp": 35.0}
        if self.stem == "bass":
            return {"brightness": 0.22, "pitch": 0.06, "hp": 20.0}
        if self.stem == "vocals":
            return {"brightness": 0.55, "pitch": 0.08, "hp": 90.0}
        return {"brightness": 0.55, "pitch": 0.08, "hp": 70.0}

    def _bar_index(self, t_sec: float, bpm: float, beats_per_bar: int) -> int:
        spb = seconds_per_beat(bpm)
        bar_s = beats_per_bar * spb
        return int(math.floor(t_sec / max(1e-9, bar_s)))

    def _ensure_motif(self, state: GlobalState, intent: Dict[str, Any], now_sec: float, beats_per_bar: int):
        subdiv = int(intent.get("subdiv", 16))
        hits = int(intent.get("hits", 6))
        mutate_bars = int(intent.get("mutate_bars", 4))
        rotate = int(intent.get("rotate", 0))

        # mutate on bar boundaries
        bi = self._bar_index(now_sec, state.tempo_bpm, beats_per_bar)
        if self._last_bar_index is None:
            self._last_bar_index = bi
        if bi != self._last_bar_index:
            # advanced by some bars; decrement counters accordingly
            adv = max(1, bi - self._last_bar_index)
            self._motif_bars_left = max(0, self._motif_bars_left - adv)
            self._last_bar_index = bi

        # if very coherent or very short samples, reduce hits (space is audible structure)
        effective_density = float(np.clip(state.density * (0.65 + 0.35 * (1.0 - state.tension)), 0.0, 1.0))
        hit_scale = (0.65 + 0.55 * effective_density) * (0.85 + 0.30 * state.energy)
        hit_scale *= (1.0 - 0.30 * state.coherence)  # coherence => fewer hits, more repetition

        hits = int(np.clip(round(hits * hit_scale), 0, subdiv))

        need_new = (self._motif_bars_left <= 0) or (subdiv != self._motif_subdiv) or (not self._motif_steps)

        if need_new:
            if self.stem == "bass":
                hits = int(np.clip(round(hits * 0.70), 1, max(1, subdiv // 2)))
            if self.stem == "vocals":
                hits = int(np.clip(round(hits * 0.55), 0, max(1, subdiv // 2)))

            patt = euclidean_rhythm(hits, subdiv)
            patt = rotate_list(patt, rotate)

            # gentle anchoring for bass/drums
            if self.stem in ("drums", "bass") and self.rng.random() < (0.55 + 0.25 * state.coherence):
                patt[0] = 1

            self._motif_steps = patt
            self._motif_subdiv = subdiv
            self._motif_bars_left = max(1, mutate_bars)
            self._motif_rotate = rotate

    def _choose_sample(
        self,
        state: GlobalState,
        allowed_clusters: List[int],
        step_strength: float,
        anchors: Optional[List[Sample]] = None,
    ) -> Optional[Sample]:
        if not self.samples:
            return None

        # repetition mechanism: probabilistic return to anchors
        anchor_prob = float(np.clip(
            0.12 + 0.78 * state.commitment * state.coherence - 0.35 * state.novelty,
            0.0, 0.90
        ))
        if anchors and self.rng.random() < anchor_prob:
            best = None
            best_score = -1e9
            for _ in range(min(12, len(anchors))):
                s = anchors[self.rng.randrange(len(anchors))]
                # allow repetition within anchors; only mild penalty
                rep_pen = self.memory.recency_penalty(self.stem, s.name)
                score = -0.28 * rep_pen + 0.02 * self.rng.random()
                if score > best_score:
                    best_score, best = score, s
            if best is not None:
                return best

        w = self._role()
        desire_bright = float(np.clip(
            0.55 * state.brightness + 0.20 * self.bias_brightness + 0.25 * w["brightness"],
            0.0, 1.0
        ))
        if self.stem in ("drums", "bass"):
            desire_bright = float(np.clip(desire_bright - 0.25 * step_strength, 0.0, 1.0))

        allowed_set = set(allowed_clusters) if allowed_clusters else None
        commitment = float(state.commitment)
        novelty = float(state.novelty)

        tries = min(28, len(self.samples))
        best, best_score = None, -1e9

        for _ in range(tries):
            s = self.samples[self.rng.randrange(len(self.samples))]
            cid = int(s.cluster_id)

            # weight toward allowed clusters; don't hard-ban
            if allowed_set is not None and cid not in allowed_set:
                if self.rng.random() < (0.20 + 0.70 * commitment):
                    continue

            f = s.features
            centroid = float(f.get("centroid", 0.0))
            bright = float(np.clip(centroid / 9000.0, 0.0, 1.0))
            bright_score = -abs(bright - desire_bright)

            rep_pen = self.memory.recency_penalty(self.stem, s.name)
            rep_score = -(0.40 + 0.55 * novelty) * rep_pen

            dur = float(f.get("dur", s.duration))
            if self.stem == "drums":
                target = 0.06 + 0.14 * (1.0 - state.tension)
            elif self.stem == "bass":
                target = 0.08 + 0.22 * (1.0 - state.tension)
            else:
                target = 0.10 + 0.35 * (1.0 - state.tension)
            dur_score = -abs(np.clip(dur, 0.02, 3.0) - target)

            score = 1.10 * bright_score + 1.35 * rep_score + 0.40 * dur_score + 0.02 * self.rng.random()
            if score > best_score:
                best_score, best = score, s

        return best if best is not None else self.samples[self.rng.randrange(len(self.samples))]

    def _render_params(self, s: Sample, state: GlobalState, step_strength: float) -> RenderParams:
        w = self._role()
        rms = float(s.features.get("rms", 0.08))

        loud = 0.045 + 0.14 * state.energy
        loud *= (1.0 + 0.12 * step_strength)
        gain = float(np.clip(loud / max(1e-4, rms), 0.02, 1.10))

        bright = float(np.clip(0.70 * state.brightness + 0.30 * w["brightness"], 0.0, 1.0))
        lp = 550.0 * (18000.0 / 550.0) ** bright
        lp = float(np.clip(lp, 250.0, 20000.0))

        hp = float(np.clip(w["hp"] + 70.0 * (1.0 - state.energy) + 40.0 * (1.0 - state.coherence), 15.0, 180.0))

        reverse_p = float(np.clip(0.01 + 0.08 * state.tension + 0.05 * state.novelty, 0.0, 0.14))
        reverse = self.rng.random() < reverse_p

        # conservative pitch for micro-samples
        depth = 0.18 * w["pitch"] + 0.10 * state.novelty
        if self.stem == "bass":
            depth *= 0.55
        if self.stem == "vocals":
            depth *= 0.65
        semitone = float(np.clip(self.rng.choice([-2, -1, 0, 0, 0, 1, 2]) * depth * 2.2, -3.0, 3.0))

        if self.stem == "drums":
            attack = 0.001
            release = 0.05 + 0.08 * (1.0 - state.tension)
        elif self.stem == "bass":
            attack = 0.003
            release = 0.09 + 0.16 * (1.0 - state.tension)
        else:
            attack = 0.004 + 0.010 * (1.0 - state.tension)
            release = 0.12 + 0.30 * (1.0 - state.tension)

        fade = float(np.clip(0.003 + 0.010 * (1.0 - state.tension), 0.002, 0.02))
        drive = float(np.clip(1.05 + 1.20 * state.energy + 0.70 * state.tension, 1.0, 3.6))

        return RenderParams(
            gain=gain,
            semitone=semitone,
            lowpass_hz=lp,
            highpass_hz=hp,
            reverse=reverse,
            drive=drive,
            fade_s=fade,
            env_attack_s=float(attack),
            env_release_s=float(release),
        )

    def plan_next(
        self,
        now_sec: float,
        state: GlobalState,
        intent: Dict[str, Any],
        allowed_clusters: List[int],
        anchors: Optional[List[Sample]],
        beats_per_bar: int = 4,
    ) -> Optional[Tuple[float, Sample, RenderParams, Dict[str, Any]]]:
        if now_sec < self.next_time_sec:
            return None
        if not self.samples:
            self.next_time_sec = now_sec + 2.0
            return None

        self._ensure_motif(state, intent, now_sec, beats_per_bar=beats_per_bar)
        subdiv = int(intent.get("subdiv", self._motif_subdiv))
        subdiv = max(1, subdiv)

        spb = seconds_per_beat(state.tempo_bpm)
        step_beats = 1.0 / float(subdiv)

        cands = quantized_time_candidates(
            now_sec=now_sec,
            bpm=state.tempo_bpm,
            subdiv=subdiv,
            swing=state.swing * (0.60 + 0.40 * state.energy),
            horizon_beats=2.0,
        )
        if not cands:
            self.next_time_sec = now_sec + 0.1
            return None

        beat_now = now_sec / spb
        step_now = int(math.floor(beat_now / step_beats))

        def step_strength(step_idx: int) -> float:
            # accent beat boundaries in 4/4
            beat_step = max(1, subdiv // beats_per_bar)
            pos = step_idx % beat_step
            if pos == 0:
                return 1.0
            if pos == beat_step // 2:
                return 0.6
            return 0.25

        # scan forward to choose next step; reduce fills when coherent
        max_scan = subdiv * 2
        chosen_step: Optional[int] = None

        # base probabilities; coherence damps randomness
        fill_p = 0.02 + 0.10 * state.novelty + 0.06 * state.tension
        fill_p *= (1.0 - 0.70 * state.coherence)

        skip_p = 0.02 + 0.08 * (1.0 - state.energy)
        skip_p *= (1.0 - 0.45 * state.coherence)

        for d in range(1, max_scan + 1):
            si = (step_now + d) % subdiv
            hit = self._motif_steps[si] if self._motif_steps else 0

            if hit == 1:
                if self.rng.random() < skip_p:
                    continue
                chosen_step = step_now + d
                break
            else:
                if self.rng.random() < fill_p:
                    chosen_step = step_now + d
                    break

        if chosen_step is None:
            self.next_time_sec = now_sec + 0.12
            return None

        target_beat = chosen_step * step_beats
        target_sec = target_beat * spb
        t = min(cands, key=lambda x: abs(x - target_sec))

        # microtiming: tiny; very coherent => almost none
        jit = (0.0015 + 0.008 * state.novelty) * (1.0 - 0.90 * state.coherence)
        t += self.rng.uniform(-jit, jit)

        # internal voice floor to avoid degenerate re-scheduling loops
        if self.stem == "drums":
            min_ioi = 0.05
        elif self.stem == "bass":
            min_ioi = 0.09
        else:
            min_ioi = 0.07
        self.next_time_sec = max(now_sec + min_ioi, t + min_ioi)

        strength = step_strength(int(chosen_step))
        s = self._choose_sample(state, allowed_clusters=allowed_clusters, step_strength=strength, anchors=anchors)
        if s is None:
            return None
        p = self._render_params(s, state, step_strength=strength)

        meta = {"subdiv": subdiv, "chosen_step": int(chosen_step), "step_strength": float(strength)}
        return (t, s, p, meta)


# -------------------- Renderer --------------------
class Renderer:
    def __init__(self, sr: int):
        self.sr = int(sr)

    def render(self, s: Sample, p: RenderParams) -> np.ndarray:
        sig = np.asarray(s.data, dtype=np.float32).copy()
        if p.reverse:
            sig = sig[::-1]

        if s.sr != self.sr:
            sig = resample_linear(sig, s.sr / self.sr)

        pr = semitone_to_ratio(p.semitone)
        if pr != 1.0:
            sig = resample_linear(sig, pr)

        sig = to_mono(sig)
        sig *= float(p.gain)

        sig = highpass_1pole(sig, p.highpass_hz, self.sr)
        sig = lowpass_1pole(sig, p.lowpass_hz, self.sr)

        env = envelope_perc(len(sig), self.sr, p.env_attack_s, p.env_release_s)
        sig *= env

        fade_samps = int(round(p.fade_s * self.sr))
        sig = apply_fade(sig, fade_samps)

        sig = soft_clip(sig, p.drive)
        return sig.astype(np.float32, copy=False)


# -------------------- Scheduling primitives --------------------
class AtomicInt:
    def __init__(self, v: int = 0):
        self._v = int(v)
        self._lock = threading.Lock()

    def get(self) -> int:
        with self._lock:
            return int(self._v)

    def set(self, v: int):
        with self._lock:
            self._v = int(v)


class Agent:
    def __init__(
        self,
        voices: List[Voice],
        sr: int,
        rng: random.Random,
        logger: RunLogger,
        dataset: DatasetProfile,
    ):
        self.voices = voices
        self.sr = int(sr)
        self.rng = rng
        self.logger = logger
        self.dataset = dataset

        self.state = GlobalState(
            tempo_bpm=float(self.rng.uniform(78.0, 142.0)),
            energy=float(self.rng.uniform(0.25, 0.85)),
            tension=float(self.rng.uniform(0.05, 0.55)),
            density=float(self.rng.uniform(0.18, 0.55)),   # start calmer
            brightness=float(self.rng.uniform(0.25, 0.75)),
            novelty=float(self.rng.uniform(0.12, 0.65)),
            coherence=float(self.rng.uniform(0.60, 0.93)),
            commitment=float(self.rng.uniform(0.35, 0.80)),  # favor repetition
            swing=float(self.rng.uniform(0.06, 0.32)),
        )
        self.state.clamp()

        self.beats_per_bar = 4

        self.critic = Critic()
        self.director = Director(self.rng)
        self.memory = voices[0].memory if voices else Memory()
        self.renderer = Renderer(sr=self.sr)

        self.palette = PaletteBag(self.rng)
        self._last_palette_refresh_bucket: Optional[int] = None

        # global density governor
        self.bucket = TokenBucket(capacity=10.0, fill_rate=5.0)

        # per-stem cooldown tracking
        self._stem_last_t: Dict[str, float] = {st: -1e9 for st in ["drums", "bass", "other", "vocals"]}

        # anchors: per-segment return pool
        self.anchors: Dict[str, List[Sample]] = {st: [] for st in ["drums", "bass", "other", "vocals"]}

        self.recent_events: List[Dict[str, Any]] = []
        self._last_state_update: float = 0.0

        self.director.maybe_transition(0.0, self.state, {})
        self.logger.start_segment(self._segment_meta(reason="init"))

        # init palette + anchors
        self._maybe_refresh_palette(force=True)
        self._refresh_anchors_for_segment()

        # GUI state
        self._lock = threading.Lock()
        self._last_critic: Dict[str, float] = {}
        self._last_event: Optional[Dict[str, Any]] = None
        self._event_count_total = 0

    def _segment_meta(self, reason: str) -> Dict[str, Any]:
        return {
            "reason": reason,
            "segment_name": self.director.name,
            "segment_ends_at": float(self.director.segment_ends_at),
            "state": {
                "tempo_bpm": self.state.tempo_bpm,
                "energy": self.state.energy,
                "tension": self.state.tension,
                "density": self.state.density,
                "brightness": self.state.brightness,
                "novelty": self.state.novelty,
                "coherence": self.state.coherence,
                "commitment": self.state.commitment,
                "swing": self.state.swing,
            },
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "time": time.time(),
                "segment": self.director.name,
                "segment_ends_at": float(self.director.segment_ends_at),
                "state": {
                    "tempo_bpm": float(self.state.tempo_bpm),
                    "energy": float(self.state.energy),
                    "tension": float(self.state.tension),
                    "density": float(self.state.density),
                    "brightness": float(self.state.brightness),
                    "novelty": float(self.state.novelty),
                    "coherence": float(self.state.coherence),
                    "commitment": float(self.state.commitment),
                    "swing": float(self.state.swing),
                },
                "critic": dict(self._last_critic),
                "events_total": int(self._event_count_total),
                "recent_events_len": int(len(self.recent_events)),
                "last_event": dict(self._last_event) if self._last_event else None,
                "bucket_tokens": float(self.bucket.tokens),
                "bucket_rate": float(self.bucket.fill_rate),
            }

    def _smooth_state_towards(self, targets: Dict[str, float], dt: float):
        a = float(np.clip(0.06 + 0.18 * dt, 0.02, 0.35))
        for k, v in targets.items():
            cur = getattr(self.state, k)
            setattr(self.state, k, float((1.0 - a) * cur + a * float(v)))
        self.state.clamp()

    def _maybe_refresh_palette(self, force: bool = False):
        b = int(np.clip(math.floor(self.state.commitment * 5), 0, 4))
        if (not force) and (self._last_palette_refresh_bucket == b):
            return
        self._last_palette_refresh_bucket = b
        for v in self.voices:
            self.palette.refresh(v.stem, v.samples, self.state.commitment)

    def _refresh_anchors_for_segment(self):
        # dynamic anchor size: higher commitment => fewer anchors => clearer repetition
        for v in self.voices:
            stem = v.stem
            allowed = set(self.palette.allowed_clusters(stem))
            pool = [s for s in v.samples if (not allowed or int(s.cluster_id) in allowed)]
            if not pool:
                self.anchors[stem] = []
                continue

            k = int(np.clip(round((1.0 - self.state.commitment) * 8 + 2), 2, 10))
            self.rng.shuffle(pool)
            self.anchors[stem] = pool[:k]

    def _update_bucket_from_state(self, critic: Dict[str, float]):
        bpm = float(self.state.tempo_bpm)
        spb = seconds_per_beat(bpm)

        chaos = float(critic.get("chaos", 0.0))

        # target "events per beat" derived from density, then damped by coherence and chaos
        ev_per_beat = 0.55 + 1.70 * float(self.state.density)
        ev_per_beat *= (1.0 - 0.55 * float(self.state.coherence))
        ev_per_beat *= (1.0 - 0.70 * chaos)

        target_rate = ev_per_beat / max(1e-6, spb)  # events/sec
        # keep it modest; short samples need a lower ceiling
        target_rate = float(np.clip(target_rate, 1.0, 7.5))

        # capacity controls burstiness: less coherent => bigger bursts allowed
        cap = 5.0 + 14.0 * (1.0 - self.state.coherence)
        cap = float(np.clip(cap, 5.0, 20.0))

        self.bucket.set_rate(fill_rate=target_rate, capacity=cap)

    def _update_state(self, now_sec: float):
        if now_sec - self._last_state_update < 0.9:
            return
        dt = now_sec - self._last_state_update
        self._last_state_update = now_sec

        critic = self.critic.analyze(self.recent_events)
        with self._lock:
            self._last_critic = dict(critic)

        self.director.maybe_transition(now_sec, self.state, critic)

        prev_name = self.logger.segments[-1]["segment_meta"].get("segment_name") if self.logger.segments else None
        if self.director.name != prev_name:
            self.logger.start_segment(self._segment_meta(reason="transition"))
            self._maybe_refresh_palette(force=True)
            self._refresh_anchors_for_segment()

        targets = self.director.current_targets()
        if targets:
            self._smooth_state_towards(targets, dt=min(2.0, dt))

        # self-correction loops (keep density sane for short samples)
        chaos = float(critic.get("chaos", 0.0))
        repetition = float(critic.get("repetition", 0.0))
        boredom = float(critic.get("boredom", 0.0))

        if chaos > 0.20:
            self.state.coherence = float(np.clip(self.state.coherence + 0.22 * chaos, 0.0, 1.0))
            self.state.density = float(np.clip(self.state.density - 0.22 * chaos, 0.0, 1.0))
            self.state.tension = float(np.clip(self.state.tension - 0.10 * chaos, 0.0, 1.0))

        if repetition > 0.25:
            # too repetitive: slightly open palette and novelty, but do NOT spike density
            self.state.novelty = float(np.clip(self.state.novelty + 0.18 * repetition, 0.0, 1.0))
            self.state.commitment = float(np.clip(self.state.commitment - 0.12 * repetition, 0.0, 1.0))

        if boredom > 0.25:
            # boredom: increase novelty/brightness a bit; small density nudge only if coherent
            self.state.novelty = float(np.clip(self.state.novelty + 0.14 * boredom, 0.0, 1.0))
            self.state.brightness = float(np.clip(self.state.brightness + 0.14 * boredom, 0.0, 1.0))
            self.state.density = float(np.clip(self.state.density + 0.10 * boredom * self.state.coherence, 0.0, 1.0))

        self.state.clamp()
        self._maybe_refresh_palette(force=False)
        self._update_bucket_from_state(critic)

    def _log_and_remember(self, e: Dict[str, Any]):
        with self._lock:
            self._last_event = dict(e)
            self._event_count_total += 1
        self.recent_events.append(e)
        if len(self.recent_events) > 1400:
            self.recent_events = self.recent_events[-900:]
        self.logger.log_event(e)
        self.memory.note(e["stem"], e["sample_name"], int(e.get("cluster_id", -1)), float(e.get("t_sec", 0.0)))

    def plan_audio_until(self, now_sample: int, until_sample: int) -> List[ScheduledAudio]:
        now_sec = now_sample / self.sr
        until_sec = until_sample / self.sr

        self._update_state(now_sec)
        self.bucket.tick(now_sec)

        out: List[ScheduledAudio] = []

        # global near-simultaneous cap (more strict when coherent)
        global_poly_cap = 2 if self.state.coherence > 0.70 else 3

        # token costs by stem (small bias, not a "style")
        stem_cost = {"drums": 1.0, "bass": 1.15, "other": 1.05, "vocals": 1.25}

        cursor_sec = now_sec
        max_events = 220

        for _ in range(max_events):
            soonest = None  # (t, voice, sample, params, meta)

            for v in self.voices:
                intent = self.director.rhythm_intent(v.stem)
                allowed = self.palette.allowed_clusters(v.stem)
                anchors = self.anchors.get(v.stem, [])
                payload = v.plan_next(
                    now_sec=cursor_sec,
                    state=self.state,
                    intent=intent,
                    allowed_clusters=allowed,
                    anchors=anchors,
                    beats_per_bar=self.beats_per_bar,
                )
                if payload is None:
                    continue
                t, s, p, meta = payload
                if t > until_sec:
                    continue

                # prevent tight pileups before even choosing soonest
                if self.memory.local_polyphony(t, window=0.040) >= global_poly_cap:
                    v.next_time_sec = max(v.next_time_sec, t + 0.02)
                    continue

                if soonest is None or t < soonest[0]:
                    soonest = (t, v, s, p, meta)

            if soonest is None:
                break

            t, v, sample, params, meta = soonest
            start_sample = int(round(t * self.sr))
            if start_sample > until_sample:
                break

            # --------- global density budget + per-stem cooldown ----------
            self.bucket.tick(t)

            # cooldown derived from tempo & coherence (not fixed ms)
            spb = seconds_per_beat(self.state.tempo_bpm)
            min_stem_gap = spb * (0.22 + 0.65 * self.state.coherence)
            if (t - self._stem_last_t.get(v.stem, -1e9)) < min_stem_gap:
                v.next_time_sec = max(v.next_time_sec, t + 0.02)
                cursor_sec = min(until_sec, t + 0.002)
                continue

            cost = float(stem_cost.get(v.stem, 1.1))
            if not self.bucket.consume(cost):
                v.next_time_sec = max(v.next_time_sec, t + 0.03)
                cursor_sec = min(until_sec, t + 0.002)
                continue

            # render + schedule
            buf = self.renderer.render(sample, params)

            sched = ScheduledAudio(
                start_sample=start_sample,
                channel=v.channel,
                buffer=buf,
                event_meta={
                    "t_sec": float(t),
                    "start_sample": int(start_sample),
                    "stem": v.stem,
                    "channel": int(v.channel),
                    "sample_name": sample.name,
                    "filename": sample.filename,
                    "cluster_id": int(sample.cluster_id),
                    "rhythm": meta,
                    "params": {
                        "gain": params.gain,
                        "semitone": params.semitone,
                        "lowpass_hz": params.lowpass_hz,
                        "highpass_hz": params.highpass_hz,
                        "reverse": params.reverse,
                        "drive": params.drive,
                        "fade_s": params.fade_s,
                        "env_attack_s": params.env_attack_s,
                        "env_release_s": params.env_release_s,
                    },
                    "state": {
                        "tempo_bpm": self.state.tempo_bpm,
                        "energy": self.state.energy,
                        "tension": self.state.tension,
                        "density": self.state.density,
                        "brightness": self.state.brightness,
                        "novelty": self.state.novelty,
                        "coherence": self.state.coherence,
                        "commitment": self.state.commitment,
                        "swing": self.state.swing,
                    },
                    "governor": {
                        "bucket_tokens": float(self.bucket.tokens),
                        "bucket_rate": float(self.bucket.fill_rate),
                        "cost": float(cost),
                        "min_stem_gap_s": float(min_stem_gap),
                        "global_poly_cap": int(global_poly_cap),
                    },
                    "segment": self.director.name,
                },
            )
            out.append(sched)
            self._log_and_remember(sched.event_meta)

            self._stem_last_t[v.stem] = float(t)
            cursor_sec = min(until_sec, t + 0.001)

        return out


class PlannerThread:
    def __init__(self, agent: Agent, current_sample: "AtomicInt", sr: int):
        self.agent = agent
        self.current_sample = current_sample
        self.sr = int(sr)

        self.heap: List[Tuple[int, int, ScheduledAudio]] = []
        self._heap_lock = threading.Lock()
        self._seq = 0

        self.lookahead_s = 0.75
        self.min_buffer_s = 0.35

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass

    def pop_ready(self, block_start: int, block_end: int) -> List[ScheduledAudio]:
        out: List[ScheduledAudio] = []
        with self._heap_lock:
            while self.heap and self.heap[0][0] < block_end:
                _, _, item = heapq.heappop(self.heap)
                out.append(item)
        return out

    def _push_many(self, items: List[ScheduledAudio]):
        with self._heap_lock:
            for it in items:
                heapq.heappush(self.heap, (int(it.start_sample), int(self._seq), it))
                self._seq += 1

    def _run(self):
        while not self._stop.is_set():
            now = self.current_sample.get()
            with self._heap_lock:
                last = self.heap[-1][0] if self.heap else now
            ahead = (last - now) / self.sr
            if ahead < self.min_buffer_s:
                until = now + int(self.lookahead_s * self.sr)
                items = self.agent.plan_audio_until(now, until)
                self._push_many(items)
            time.sleep(0.01)


# -------------------- GUI -----------------------------
class StateMonitorGUI:
    def __init__(self, root: "tk.Tk", agent: Agent, on_close: Callable[[], None], refresh_ms: int = 200):
        self.root = root
        self.agent = agent
        self.on_close = on_close
        self.refresh_ms = int(refresh_ms)

        root.title("Groove Agentic Monitor")
        root.geometry("560x600")

        self.text = tk.Text(root, wrap="word", height=34)
        self.text.pack(fill="both", expand=True)

        btn = ttk.Button(root, text="Stop", command=self._handle_close)
        btn.pack(fill="x")

        root.protocol("WM_DELETE_WINDOW", self._handle_close)
        self._tick()

    def _handle_close(self):
        try:
            self.on_close()
        finally:
            self.root.destroy()

    def _tick(self):
        snap = self.agent.snapshot()
        now = snap["time"]
        seg = snap["segment"]
        ends = snap["segment_ends_at"]
        remaining = max(0.0, ends - now) if ends else 0.0

        lines = []
        lines.append(f"Segment: {seg}   (ends in ~{remaining:.1f}s)")
        lines.append(f"Governor: tokens={snap.get('bucket_tokens', 0.0):.2f}  rate={snap.get('bucket_rate', 0.0):.2f}/s")
        lines.append("")
        lines.append("State:")
        for k, v in snap["state"].items():
            lines.append(f"  {k:>10}: {v:.3f}" if isinstance(v, float) else f"  {k:>10}: {v}")

        lines.append("")
        lines.append("Critic:")
        if snap["critic"]:
            for k, v in snap["critic"].items():
                lines.append(f"  {k:>12}: {v:.3f}" if isinstance(v, float) else f"  {k:>12}: {v}")
        else:
            lines.append("  (no data yet)")

        lines.append("")
        lines.append(f"Events total: {snap['events_total']}   Recent buffer: {snap['recent_events_len']}")

        last = snap["last_event"]
        lines.append("")
        lines.append("Last event:")
        if last:
            lines.append(f"  stem: {last.get('stem')}  ch: {last.get('channel')}  cluster: {last.get('cluster_id')}")
            lines.append(f"  sample: {last.get('sample_name')}")
            r = last.get("rhythm", {})
            lines.append(f"  rhythm: subdiv={r.get('subdiv')} step={r.get('chosen_step')} strength={r.get('step_strength')}")
            g = last.get("governor", {})
            lines.append(f"  governor: cost={g.get('cost')} min_gap={g.get('min_stem_gap_s'):.3f}s poly_cap={g.get('global_poly_cap')}")
        else:
            lines.append("  (none yet)")

        self.text.delete("1.0", "end")
        self.text.insert("1.0", "\n".join(lines))
        self.root.after(self.refresh_ms, self._tick)


# -------------------- Audio engine --------------------
class MasterLimiter:
    def __init__(self, target: float = 0.92, attack: float = 0.02, release: float = 0.995):
        self.target = float(target)
        self.attack = float(attack)
        self.release = float(release)
        self.gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x)) + 1e-12)
        desired = min(1.0, self.target / peak)
        if desired < self.gain:
            self.gain = (1.0 - self.attack) * self.gain + self.attack * desired
        else:
            self.gain = self.release * self.gain + (1.0 - self.release) * desired
        return (x * self.gain).astype(np.float32, copy=False)


class AudioEngine:
    def __init__(self, planner: PlannerThread, current_sample: AtomicInt, sr: int, channels: int, blocksize: int, latency: str):
        self.planner = planner
        self.current_sample = current_sample
        self.sr = int(sr)
        self.channels = int(channels)
        self.blocksize = int(blocksize)

        self.active: List[PlayEvent] = []
        self.active_lock = threading.Lock()

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.channels,
            dtype="float32",
            latency=latency,
            callback=self.callback,
        )
        self.limiter = MasterLimiter(target=0.92)

    def start(self):
        self.stream.start()

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

    def _activate(self, sched: ScheduledAudio, block_start: int):
        delay = max(0, int(sched.start_sample - block_start))
        pe = PlayEvent(buffer=sched.buffer, channel=int(sched.channel), pos=0, delay=delay)
        with self.active_lock:
            self.active.append(pe)

    def _mix_active_into(self, out: np.ndarray, start: int, end: int):
        length = end - start
        with self.active_lock:
            new_active: List[PlayEvent] = []
            for ev in self.active:
                if ev.channel >= out.shape[1]:
                    continue

                seg_start = start
                seg_len = length
                if ev.delay > 0:
                    consume = min(seg_len, ev.delay)
                    ev.delay -= consume
                    if consume == seg_len:
                        new_active.append(ev)
                        continue
                    seg_start += consume
                    seg_len -= consume

                remaining = ev.buffer.shape[0] - ev.pos
                if remaining <= 0:
                    continue

                n = min(seg_len, remaining)
                out[seg_start:seg_start + n, ev.channel] += ev.buffer[ev.pos:ev.pos + n]
                ev.pos += n

                if ev.pos < ev.buffer.shape[0]:
                    new_active.append(ev)
            self.active = new_active

    def callback(self, outdata, frames, time_info, status):
        block_start = self.current_sample.get()
        block_end = block_start + frames
        self.current_sample.set(block_end)

        out = np.zeros((frames, self.channels), dtype=np.float32)

        ready = self.planner.pop_ready(block_start, block_end)
        for sched in ready:
            self._activate(sched, block_start)

        self._mix_active_into(out, 0, frames)

        out = np.tanh(out * 0.95).astype(np.float32)
        out = self.limiter.process(out)
        np.clip(out, -1.0, 1.0, out)
        outdata[:] = out


# -------------------- Build system --------------------
def build_system(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    rng = random.Random(args.seed)

    if not os.path.exists(args.db_path):
        print(f"Error: Database file '{args.db_path}' not found!")
        sys.exit(1)
    if not os.path.exists(args.samples_dir):
        print(f"Error: Samples directory '{args.samples_dir}' not found!")
        sys.exit(1)

    db = SampleDatabase(args.db_path, args.samples_dir, rng)

    stems = ["drums", "bass", "other", "vocals"]
    samples_by_stem: Dict[str, List[Sample]] = {}
    for st in stems:
        lst = db.load_by_stem(st, args.samples_limit)
        samples_by_stem[st] = lst
        print(f"Loaded {len(lst)} {st} samples")
    db.close()

    total = sum(len(v) for v in samples_by_stem.values())
    if total == 0:
        print("Error: No samples loaded.")
        sys.exit(1)

    fx = FeatureExtractor(sr_target=args.sr)
    for lst in samples_by_stem.values():
        for s in lst:
            try:
                s.features = fx.extract(s)
            except Exception:
                s.features = {}

    Clusterer(rng).assign(samples_by_stem)
    dataset = DatasetProfile(samples_by_stem)

    logger = RunLogger(
        run_meta={
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": args.seed,
            "audio": {"sr": args.sr, "blocksize": args.blocksize, "channels": args.channels, "latency": args.latency},
            "db": {"db_path": args.db_path, "samples_dir": args.samples_dir, "samples_limit": args.samples_limit},
            "system": "groove_first_rewrite_with_governor_anchors",
        }
    )

    memory = Memory(per_voice=180)
    cmap = channel_map_for(args.channels)

    voices: List[Voice] = []
    for st in stems:
        vrng = random.Random(rng.randint(0, 2**30))
        voices.append(Voice(stem=st, channel=cmap[st], samples=samples_by_stem[st], rng=vrng, memory=memory))

    current_sample = AtomicInt(0)
    agent = Agent(voices=voices, sr=args.sr, rng=rng, logger=logger, dataset=dataset)
    planner = PlannerThread(agent=agent, current_sample=current_sample, sr=args.sr)
    engine = AudioEngine(planner=planner, current_sample=current_sample, sr=args.sr, channels=args.channels,
                         blocksize=args.blocksize, latency=args.latency)

    return logger, planner, engine, agent


# -------------------- Main --------------------
def main():
    args = parse_arguments()
    logger, planner, engine, agent = build_system(args)

    stopping = threading.Event()

    try:
        planner.start()
        engine.start()

        if _TK_AVAILABLE:
            root = tk.Tk()

            def request_stop():
                stopping.set()
                try:
                    root.quit()
                except Exception:
                    pass

            StateMonitorGUI(root, agent, on_close=request_stop, refresh_ms=200)
            root.mainloop()
        else:
            print("GUI not available (tkinter import failed). Running headless. Press Ctrl+C to stop.")
            while not stopping.is_set():
                time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        engine.stop()
        planner.stop()
        path = logger.finalize()
        print(f"Run log written: {path}")


if __name__ == "__main__":
    main()
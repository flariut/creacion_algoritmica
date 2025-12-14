#!/usr/bin/env python3
"""
Nueva versión del sequencer.
Casi todos los parámetros avanzan con random walks.
Se agrega logueo de eventos directo a disco,
saturación y limitador de salida para más placer.
"""

import os
import sys
import math
import random
import threading
import sqlite3
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from collections import deque

import numpy as np
import soundfile as sf
import sounddevice as sd

try:
    import yaml
except ImportError:
    yaml = None

# ---------------- YAML Configuration ----------------

DEFAULT_CONFIG_YAML = """\
db_path: "microsamples.db"
samples_dir: "samples"

audio:
  sr: 44100
  blocksize: 4096
  channels: 1          # 1=mono, 2=stereo, 4=quad
  latency: "high"

sequencer:
  bpm: 120.0
  steps: 16
  repeat_min: 4
  repeat_max: 16
  reverse_prob: 0.2
  samples_per_track: 1000

processing:
  fade_duration: 0.005

  # AHD (Attack / Hold / Decay) envelope per-trigger
  # Razonable para microsamples ~500ms:
  # - Attack: corto (2..60ms)
  # - Hold: 0..250ms
  # - Decay: 20..450ms
  # Se clampa al largo real del buffer ya resampleado.
  envelope:
    enabled: true
    attack_ms_min: 2.0
    attack_ms_max: 60.0
    hold_ms_min: 0.0
    hold_ms_max: 250.0
    decay_ms_min: 20.0
    decay_ms_max: 450.0

stack:
  enabled: true
  prob: 0.05
  max: 64
  decay_steps: 128

clip:
  mode: "tube"         # "tube" (asym tanh) or "tanh" (simple tanh)
  drive: 1.0
  asym: -0.3

limiter:
  enabled: true
  threshold: 0.98      # peak ceiling
  attack_ms: 2.0
  release_ms: 80.0
  pre_gain: 3.0

random:
  seed: null

drift:
  enabled: true

  # applied each pattern regeneration (random-walk style)
  bpm_rate: 10.0        # +/- bpm
  bpm_min: 40.0
  bpm_max: 220.0

  steps_rate: 2         # +/- steps (integer)
  steps_min: 4
  steps_max: 64

  repeat_min_rate: 1    # +/- (integer)
  repeat_max_rate: 2    # +/- (integer)
  repeat_min_floor: 1
  repeat_max_ceil: 128

  density_rate: 0.20    # +/- density
  density_min: 0.01
  density_max: 0.99

  # intensity is COARSE random-walk clamped to [1, 5]
  intensity_regen_min: 1.0
  intensity_regen_max: 5.0
  intensity_rate: 0.6   # +/- per regeneration (random-walk)

  # small per-step variation: multiplier = clamp(1 + U(-j, +j), >=0)
  # j=1.0 => multiplier range [0, 2]
  intensity_step_jitter: 1.0
"""


def _deep_merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    if yaml is None:
        print("Error: PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)

    cfg_path = Path(path)
    if not cfg_path.exists():
        cfg_path.write_text(DEFAULT_CONFIG_YAML, encoding="utf-8")
        print(f"Wrote default config to: {cfg_path.resolve()}")
        print("Edit it if needed, then run again.")
        sys.exit(0)

    default_cfg = yaml.safe_load(DEFAULT_CONFIG_YAML)
    user_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return _deep_merge(default_cfg, user_cfg)


def cfg_get(cfg: Dict[str, Any], path: str, default=None):
    cur = cfg
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ---------------- Utils ----------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(x)))


# ---------------- Channel Mapping ----------------

def get_channel_mapping(output_channels: int) -> Dict[str, int]:
    if output_channels == 1:
        return {"drums": 0, "bass": 0, "other": 0, "vocals": 0}
    if output_channels == 2:
        return {"drums": 0, "bass": 0, "other": 1, "vocals": 1}
    return {"drums": 0, "bass": 1, "other": 2, "vocals": 3}


# ---------------- DSP helpers -------------------

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


def apply_gain(signal: np.ndarray, gain: float) -> np.ndarray:
    return signal * gain


def apply_fade(signal: np.ndarray, fade_samples: int) -> np.ndarray:
    if len(signal) == 0 or fade_samples <= 0:
        return signal

    if len(signal) <= fade_samples * 2:
        fade_len = min(fade_samples, len(signal) // 2)
        if fade_len <= 0:
            return signal
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        if signal.ndim == 1:
            signal[:fade_len] *= fade_in
            signal[-fade_len:] *= fade_out
        else:
            for c in range(signal.shape[1]):
                signal[:fade_len, c] *= fade_in
                signal[-fade_len:, c] *= fade_out
        return signal

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


# ----------- AHD Envelope (per-trigger) -----------

def _ms_to_samples(ms: float, sr: int) -> int:
    return max(0, int(round(float(ms) * sr / 1000.0)))


def random_ahd_envelope_ms(
    rng: random.Random,
    total_ms: float,
    attack_ms_min: float,
    attack_ms_max: float,
    hold_ms_min: float,
    hold_ms_max: float,
    decay_ms_min: float,
    decay_ms_max: float,
):
    total_ms = max(0.0, float(total_ms))
    if total_ms <= 0.0:
        return 0.0, 0.0, 0.0

    a_lo = max(0.0, float(attack_ms_min))
    a_hi = max(a_lo, float(attack_ms_max))
    h_lo = max(0.0, float(hold_ms_min))
    h_hi = max(h_lo, float(hold_ms_max))
    d_lo_cfg = max(0.0, float(decay_ms_min))
    d_hi_cfg = max(d_lo_cfg, float(decay_ms_max))

    # Attack
    a_hi_eff = min(a_hi, total_ms)
    if a_hi_eff < a_lo:
        a = a_hi_eff
    elif a_hi_eff == a_lo:
        a = a_lo
    else:
        a = rng.uniform(a_lo, a_hi_eff)

    remaining = max(0.0, total_ms - a)

    # Decay
    d_hi_eff = min(d_hi_cfg, remaining)
    d_lo_eff = min(d_lo_cfg, d_hi_eff)
    if d_hi_eff < d_lo_eff:
        d = d_hi_eff
    elif d_hi_eff == d_lo_eff:
        d = d_lo_eff
    else:
        d = rng.uniform(d_lo_eff, d_hi_eff)

    remaining2 = max(0.0, total_ms - a - d)

    # Hold  (FIXED HERE)
    h_hi_eff = min(h_hi, remaining2)
    if h_hi_eff < h_lo:
        h = h_hi_eff
    elif h_hi_eff == h_lo:
        h = h_lo
    else:
        h = rng.uniform(h_lo, h_hi_eff)

    # final clamp (safety)
    s = a + h + d
    if s > total_ms:
        overflow = s - total_ms
        dh = min(h, overflow); h -= dh; overflow -= dh
        if overflow > 0:
            dd = min(d, overflow); d -= dd; overflow -= dd
        if overflow > 0:
            da = min(a, overflow); a -= da

    return max(0.0, a), max(0.0, h), max(0.0, d)


def apply_ahd_envelope(signal: np.ndarray, sr: int, attack_ms: float, hold_ms: float, decay_ms: float) -> np.ndarray:
    """
    Click-safe AHD:
      - attack: cosine ramp 0->1 (includes both endpoints)
      - hold:   1
      - decay:  cosine ramp 1->0 (includes both endpoints, ends at EXACT 0)
    Remaining tail stays at 0.
    """
    if signal.size == 0:
        return signal

    n = int(signal.shape[0])
    if n <= 0:
        return signal

    a_n = _ms_to_samples(attack_ms, sr)
    h_n = _ms_to_samples(hold_ms, sr)
    d_n = _ms_to_samples(decay_ms, sr)

    # Clamp total length to buffer length (reduce hold first, then decay, then attack)
    total = a_n + h_n + d_n
    if total > n:
        overflow = total - n
        dh = min(h_n, overflow); h_n -= dh; overflow -= dh
        if overflow > 0:
            dd = min(d_n, overflow); d_n -= dd; overflow -= dd
        if overflow > 0:
            da = min(a_n, overflow); a_n -= da

    env = np.zeros((n,), dtype=np.float32)
    pos = 0

    # Attack (cosine from 0 to 1)
    if a_n > 0:
        if a_n == 1:
            env[pos] = 1.0
            pos += 1
        else:
            t = np.linspace(0.0, 1.0, a_n, endpoint=True, dtype=np.float32)
            env[pos:pos + a_n] = 0.5 - 0.5 * np.cos(np.pi * t)  # 0..1
            pos += a_n

    # Hold
    if h_n > 0:
        env[pos:pos + h_n] = 1.0
        pos += h_n

    # Decay (cosine from 1 to 0) -> ensures last decay sample is exactly 0
    if d_n > 0:
        if d_n == 1:
            env[pos] = 0.0
            pos += 1
        else:
            t = np.linspace(0.0, 1.0, d_n, endpoint=True, dtype=np.float32)
            env[pos:pos + d_n] = 0.5 + 0.5 * np.cos(np.pi * t)  # 1..0
            pos += d_n

    if signal.ndim == 1:
        signal *= env
    else:
        signal *= env[:, None]
    return signal


def _soft_clip_tube_channel(x: np.ndarray, drive: float, asym: float) -> np.ndarray:
    y = x * drive
    pos_scale = 1.0 + max(0.0, asym)
    neg_scale = 1.0 + max(0.0, -asym)
    pos = np.tanh(pos_scale * np.maximum(0.0, y))
    neg = np.tanh(neg_scale * np.minimum(0.0, y))
    return pos + neg


def apply_soft_clip(out: np.ndarray, mode: str, drive: float, asym: float) -> np.ndarray:
    mode = (mode or "tube").lower().strip()
    if mode == "tanh":
        out[:] = np.tanh(out * drive)
    else:
        for ch in range(out.shape[1]):
            out[:, ch] = _soft_clip_tube_channel(out[:, ch], drive, asym)

    np.clip(out, -1.0, 1.0, out)
    return out


class MasterLimiter:
    """
    Lightweight peak limiter (no lookahead) with attack/release smoothing.
    """
    def __init__(
        self,
        sr: int,
        threshold: float = 0.98,
        attack_ms: float = 2.0,
        release_ms: float = 80.0,
        pre_gain: float = 1.0,
        instant_catch: bool = True,
    ):
        self.sr = int(sr)
        self.threshold = float(threshold)
        self.attack_ms = float(attack_ms)
        self.release_ms = float(release_ms)
        self.pre_gain = float(pre_gain)
        self.instant_catch = bool(instant_catch)
        self.gain = 1.0

    def _coeff(self, ms: float, frames: int) -> float:
        ms = max(0.1, float(ms))
        frames = max(1, int(frames))
        t = ms / 1000.0
        return math.exp(-float(frames) / (t * self.sr))

    def process(self, x: np.ndarray, frames: Optional[int] = None) -> np.ndarray:
        if x.size == 0:
            return x

        x *= self.pre_gain

        if frames is None:
            frames = int(x.shape[0]) if x.ndim >= 1 else 1
        frames = max(1, int(frames))

        thr = max(1e-6, self.threshold)
        peak = float(np.max(np.abs(x)))
        if peak <= 0.0:
            return x

        target_gain = 1.0 if peak <= thr else (thr / peak)

        if self.instant_catch and target_gain < 1.0:
            if target_gain < self.gain:
                self.gain = target_gain

        if target_gain < self.gain:
            c = self._coeff(self.attack_ms, frames)
        else:
            c = self._coeff(self.release_ms, frames)

        self.gain = c * self.gain + (1.0 - c) * target_gain
        x *= self.gain
        return x


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
        self.rng.shuffle(rows)

        if limit is not None:
            rows = rows[:limit]

        samples: List[Sample] = []
        for filename, stem_type_row, position, duration, metadata_json in rows:
            audio_path = os.path.join(self.samples_dir, filename)
            try:
                if not os.path.exists(audio_path):
                    print(f"Warning: Sample file not found: {audio_path}")
                    continue
                data, sr = sf.read(audio_path, always_2d=False)
                data = np.asarray(data, dtype=np.float32)
                metadata = json.loads(metadata_json) if metadata_json else {}
                samples.append(
                    Sample(
                        data=data,
                        sr=sr,
                        name=f"{metadata.get('artist', 'Unknown')} - {metadata.get('song_title', 'Unknown')} - {stem_type_row}",
                        stem_type=stem_type_row,
                        original_position=position,
                        rms=metadata.get("rms", 0.5),
                        duration=float(duration) if duration is not None else float(len(data) / sr),
                    )
                )
            except Exception as e:
                print(f"Error loading sample {filename}: {e}")
                continue

        print(f"Loaded {len(samples)} {stem_type} samples from {self.samples_dir}")
        return samples

    def close(self):
        self.conn.close()


def load_samples_from_db(db_path: str, samples_dir: str, samples_per_track: int, rng: random.Random) -> Dict[str, List[Sample]]:
    db = SampleDatabase(db_path, samples_dir, rng)
    samples_by_type: Dict[str, List[Sample]] = {}
    for stem_type in ["drums", "bass", "other", "vocals"]:
        samples_by_type[stem_type] = db.get_samples_by_stem_type(stem_type, samples_per_track)
        if not samples_by_type[stem_type]:
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

    # Per-trigger AHD envelope params (ms); these are set at trigger-time (variability)
    attack_ms: float = 0.0
    hold_ms: float = 0.0
    decay_ms: float = 0.0


@dataclass
class Pattern:
    steps: List[Step] = field(default_factory=list)

    @classmethod
    def random(cls, steps: int, sample_count: int, rng: random.Random, pattern_density: float, pattern_intensity: float):
        p = cls()
        for i in range(steps):
            p.steps.append(
                Step(
                    index=i,
                    sample_idx=rng.randrange(max(1, sample_count)),
                    on=(rng.random() < pattern_density),
                    prob=rng.uniform(0.1, 1.0),
                    semitone=0.0,
                    gain=rng.uniform(0.1 * pattern_intensity, 0.2 * pattern_intensity),
                    lowpass=20000.0,
                    attack_ms=0.0,
                    hold_ms=0.0,
                    decay_ms=0.0,
                )
            )
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
    stack_ages: Dict[int, int] = field(default_factory=dict)
    stack_lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class PlayEvent:
    buffer: np.ndarray
    channel: int
    pos: int = 0


# ---------------- Clock -----------------

class Clock:
    """Sample-accurate clock emitting step boundaries and a monotonic step counter."""
    def __init__(self, bpm: float, sr: int, steps_per_beat: int = 4):
        self.sr = sr
        self.steps_per_beat = steps_per_beat
        self.sample_pos = 0
        self.step_counter = 0
        self.bpm = float(bpm)
        self.samples_per_step = 0
        self.next_step_sample = 0
        self.set_bpm(self.bpm, reset_phase=True)

    def set_bpm(self, bpm: float, reset_phase: bool = False):
        self.bpm = float(bpm)
        quarter_samples = self.sr * 60.0 / max(1e-9, self.bpm)
        self.samples_per_step = max(1, int(round(quarter_samples / self.steps_per_beat)))
        if reset_phase:
            self.sample_pos = 0
            self.step_counter = 0
            self.next_step_sample = self.samples_per_step
        else:
            self.next_step_sample = self.sample_pos + self.samples_per_step

    def advance(self, frames: int) -> List[Tuple[int, int]]:
        events: List[Tuple[int, int]] = []
        end = self.sample_pos + frames
        while self.next_step_sample <= end:
            events.append((self.next_step_sample, self.step_counter))
            self.step_counter += 1
            self.next_step_sample += self.samples_per_step
        self.sample_pos = end
        return events


# ---------------- Run logging (streaming, no RAM growth) -----------------

class RunLogger:
    """
    Streams run data to disk as NDJSON lines (JSONL):
      - meta.json
      - patterns.jsonl
      - events.jsonl
    """
    def __init__(self, run_meta: Dict[str, Any], runs_dir: Path = Path("runs")):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.runs_dir / f"run_{ts}"
        self.run_dir.mkdir(exist_ok=True)

        self.meta_path = self.run_dir / "meta.json"
        self.patterns_path = self.run_dir / "patterns.jsonl"
        self.events_path = self.run_dir / "events.jsonl"

        run_meta = dict(run_meta)
        run_meta.setdefault("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.meta_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

        self._pat_f = open(self.patterns_path, "a", encoding="utf-8", buffering=1)
        self._evt_f = open(self.events_path, "a", encoding="utf-8", buffering=1)

        self.pattern_index = 0
        self.current_pattern_index = None

        print(f"[RunLogger] writing to: {self.run_dir.resolve()}")

    def _snapshot_tracks_pattern(self, sequencer: "Sequencer") -> List[Dict[str, Any]]:
        tracks_meta: List[Dict[str, Any]] = []
        for t in sequencer.tracks:
            steps_meta = [
                {
                    "index": int(s.index),
                    "sample_idx": int(s.sample_idx),
                    "on": bool(s.on),
                    "prob": float(s.prob),
                    "semitone": float(s.semitone),
                    "gain": float(s.gain),
                    "lowpass": float(s.lowpass),
                }
                for s in t.pattern.steps
            ]
            tracks_meta.append(
                {
                    "track_id": int(t.id),
                    "stem_type": t.stem_type,
                    "channel": int(t.channel),
                    "sample_count": int(len(t.samples)),
                    "steps": steps_meta,
                }
            )
        return tracks_meta

    def start_pattern(self, sequencer: "Sequencer", reason: str):
        pat = {
            "pattern_index": int(self.pattern_index),
            "reason": str(reason),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "bpm": float(sequencer.bpm),
            "steps": int(sequencer.steps),
            "repeat_min": int(sequencer.repeat_min),
            "repeat_max": int(sequencer.repeat_max),
            "loop_target": int(sequencer.loop_target),
            "density": float(sequencer.density),
            "intensity_base": float(sequencer.intensity_base),
            "tracks": self._snapshot_tracks_pattern(sequencer),
        }
        self._pat_f.write(json.dumps(pat) + "\n")
        self.current_pattern_index = self.pattern_index
        self.pattern_index += 1

    def log_event(self, ev: Dict[str, Any]):
        if self.current_pattern_index is None:
            self.current_pattern_index = -1
        ev = dict(ev)
        ev["pattern_index"] = int(self.current_pattern_index)
        self._evt_f.write(json.dumps(ev) + "\n")

    def finalize(self):
        try:
            self._pat_f.close()
        finally:
            self._evt_f.close()
        return self.run_dir


# ---------------- Sequencer -----------------

class Sequencer:
    def __init__(self, samples_by_type: Dict[str, List[Sample]], cfg: Dict[str, Any], rng: random.Random, run_logger: RunLogger):
        self.samples_by_type = samples_by_type
        self.cfg = cfg
        self.rng = rng
        self.run_logger = run_logger

        self.sr = int(cfg_get(cfg, "audio.sr"))
        self.channels = int(cfg_get(cfg, "audio.channels"))
        self.fade_samples = int(float(cfg_get(cfg, "processing.fade_duration")) * self.sr)

        self.reverse_prob = _clamp(float(cfg_get(cfg, "sequencer.reverse_prob")), 0.0, 1.0)

        # envelope config
        self.env_enabled = bool(cfg_get(cfg, "processing.envelope.enabled", True))
        self.env_attack_ms_min = float(cfg_get(cfg, "processing.envelope.attack_ms_min", 2.0))
        self.env_attack_ms_max = float(cfg_get(cfg, "processing.envelope.attack_ms_max", 60.0))
        self.env_hold_ms_min = float(cfg_get(cfg, "processing.envelope.hold_ms_min", 0.0))
        self.env_hold_ms_max = float(cfg_get(cfg, "processing.envelope.hold_ms_max", 250.0))
        self.env_decay_ms_min = float(cfg_get(cfg, "processing.envelope.decay_ms_min", 20.0))
        self.env_decay_ms_max = float(cfg_get(cfg, "processing.envelope.decay_ms_max", 450.0))

        self.stack_enabled = bool(cfg_get(cfg, "stack.enabled"))
        self.stack_prob = float(cfg_get(cfg, "stack.prob"))
        self.stack_max = max(1, int(cfg_get(cfg, "stack.max")))
        self.stack_decay_steps = max(1, int(cfg_get(cfg, "stack.decay_steps")))

        # drift config
        self.drift_enabled = bool(cfg_get(cfg, "drift.enabled", True))

        self.bpm_rate = float(cfg_get(cfg, "drift.bpm_rate"))
        self.bpm_min = float(cfg_get(cfg, "drift.bpm_min"))
        self.bpm_max = float(cfg_get(cfg, "drift.bpm_max"))

        self.steps_rate = int(cfg_get(cfg, "drift.steps_rate"))
        self.steps_min = int(cfg_get(cfg, "drift.steps_min"))
        self.steps_max = int(cfg_get(cfg, "drift.steps_max"))

        self.repeat_min_rate = int(cfg_get(cfg, "drift.repeat_min_rate"))
        self.repeat_max_rate = int(cfg_get(cfg, "drift.repeat_max_rate"))
        self.repeat_min_floor = int(cfg_get(cfg, "drift.repeat_min_floor"))
        self.repeat_max_ceil = int(cfg_get(cfg, "drift.repeat_max_ceil"))

        self.density_rate = float(cfg_get(cfg, "drift.density_rate"))
        self.density_min = float(cfg_get(cfg, "drift.density_min"))
        self.density_max = float(cfg_get(cfg, "drift.density_max"))

        self.intensity_min = float(cfg_get(cfg, "drift.intensity_regen_min"))
        self.intensity_max = float(cfg_get(cfg, "drift.intensity_regen_max"))
        self.intensity_rate = float(cfg_get(cfg, "drift.intensity_rate", 0.6))
        self.intensity_step_jitter = float(cfg_get(cfg, "drift.intensity_step_jitter"))

        # mutable sequencer state
        self.bpm = float(cfg_get(cfg, "sequencer.bpm"))
        self.steps = int(cfg_get(cfg, "sequencer.steps"))
        self.repeat_min = int(cfg_get(cfg, "sequencer.repeat_min"))
        self.repeat_max = int(cfg_get(cfg, "sequencer.repeat_max"))

        self.density = _clamp(self.rng.random(), self.density_min, self.density_max)
        self.intensity_base = _clamp(self.rng.uniform(self.intensity_min, self.intensity_max), self.intensity_min, self.intensity_max)

        # tracks
        self.channel_map = get_channel_mapping(self.channels)
        self.tracks: List[Track] = []
        for i, stem_type in enumerate(["drums", "bass", "other", "vocals"]):
            samples = samples_by_type.get(stem_type, [])
            track_rng = random.Random(self.rng.randint(0, 2**30))
            tr = Track(
                id=i,
                stem_type=stem_type,
                channel=self.channel_map[stem_type],
                samples=samples,
                pattern=Pattern(),
                rng=track_rng,
            )
            tr.played_stack = deque(maxlen=self.stack_max)
            tr.stack_ages = {}
            self.tracks.append(tr)

        # loop control
        self.loop_count = 0
        self.loop_target = 1

        self.create_pattern_block(reason="init", drift=False)

    def _print_pattern_stats(self, reason: str):
        per_track = ", ".join(f"{t.stem_type}:samples={len(t.samples)},ch={t.channel}" for t in self.tracks)
        print(
            "\n".join(
                [
                    "",
                    f"[PatternBlock] reason={reason}",
                    f"  bpm={self.bpm:.3f}  steps={self.steps}",
                    f"  repeat_min={self.repeat_min}  repeat_max={self.repeat_max}  loop_target={self.loop_target}",
                    f"  density={self.density:.3f}",
                    f"  intensity_base={self.intensity_base:.3f} (coarse RW [{self.intensity_min:.1f},{self.intensity_max:.1f}] rate=±{self.intensity_rate:.3f})",
                    f"  intensity_step_jitter=±{self.intensity_step_jitter:.3f}  (mult = clamp(1+U(-j,+j),>=0))",
                    f"  reverse_prob={self.reverse_prob:.3f}",
                    f"  env_enabled={self.env_enabled}  A=[{self.env_attack_ms_min:.1f},{self.env_attack_ms_max:.1f}]ms  "
                    f"H=[{self.env_hold_ms_min:.1f},{self.env_hold_ms_max:.1f}]ms  D=[{self.env_decay_ms_min:.1f},{self.env_decay_ms_max:.1f}]ms",
                    f"  stack_enabled={self.stack_enabled}  stack_prob={self.stack_prob:.3f}  stack_max={self.stack_max}  stack_decay_steps={self.stack_decay_steps}",
                    f"  tracks: {per_track}",
                    "",
                ]
            )
        )

    def create_pattern_block(self, reason: str, drift: bool):
        if drift and self.drift_enabled:
            self.bpm = _clamp(self.bpm + self.rng.uniform(-self.bpm_rate, self.bpm_rate), self.bpm_min, self.bpm_max)
            self.steps = _clamp_int(self.steps + self.rng.randint(-self.steps_rate, self.steps_rate), self.steps_min, self.steps_max)

            self.repeat_min = _clamp_int(
                self.repeat_min + self.rng.randint(-self.repeat_min_rate, self.repeat_min_rate),
                self.repeat_min_floor,
                self.repeat_max_ceil,
            )
            self.repeat_max = _clamp_int(
                self.repeat_max + self.rng.randint(-self.repeat_max_rate, self.repeat_max_rate),
                self.repeat_min_floor,
                self.repeat_max_ceil,
            )
            if self.repeat_max < self.repeat_min:
                self.repeat_max = self.repeat_min

            self.density = _clamp(
                self.density + self.rng.uniform(-self.density_rate, self.density_rate),
                self.density_min,
                self.density_max,
            )

            self.intensity_base = _clamp(
                self.intensity_base + self.rng.uniform(-self.intensity_rate, self.intensity_rate),
                self.intensity_min,
                self.intensity_max,
            )

        for track in self.tracks:
            track.pattern = Pattern.random(
                steps=self.steps,
                sample_count=max(1, len(track.samples)),
                rng=track.rng,
                pattern_density=self.density,
                pattern_intensity=self.intensity_base,
            )

        self.loop_target = self.rng.randint(self.repeat_min, self.repeat_max)
        self.loop_count = 0

        self.run_logger.start_pattern(self, reason=reason)
        self._print_pattern_stats(reason=reason)

    def _step_intensity_multiplier(self) -> float:
        j = max(0.0, float(self.intensity_step_jitter))
        return max(0.0, 1.0 + self.rng.uniform(-j, j))

    def decide_for_step(self, global_step: int, step_index: int) -> List[Tuple["Track", Sample, bool, bool, Step, float]]:
        results: List[Tuple[Track, Sample, bool, bool, Step, float]] = []
        intensity_mult = self._step_intensity_multiplier()

        for track in self.tracks:
            step = track.pattern.steps[step_index]
            if (not step.on) or (track.rng.random() >= step.prob) or (not track.samples):
                continue

            used_stack = False
            chosen_sample: Optional[Sample] = None

            if self.stack_enabled and (track.rng.random() < self.stack_prob) and len(track.played_stack) > 0:
                with track.stack_lock:
                    n = len(track.played_stack)
                    weights = [(idx + 1) for idx in range(n)]  # newest heavier
                    total = sum(weights)
                    r = track.rng.random() * total
                    acc = 0.0
                    chosen_idx = 0
                    for idx, w in enumerate(weights):
                        acc += w
                        if r <= acc:
                            chosen_idx = idx
                            break
                    chosen_sample_idx = list(track.played_stack)[chosen_idx]
                    chosen_sample = track.samples[chosen_sample_idx]
                    try:
                        track.played_stack.remove(chosen_sample_idx)
                        track.stack_ages.pop(chosen_sample_idx, None)
                    except ValueError:
                        pass
                    used_stack = True

            if not used_stack:
                samp_idx = step.sample_idx % len(track.samples)
                chosen_sample = track.samples[samp_idx]

            reversed_play = (track.rng.random() < self.reverse_prob)

            # Per-trigger envelope (AHD), generated here so it gets logged with the event.
            if self.env_enabled:
                # We assume microsamples are ~500ms, but clamp to actual sample duration.
                # Use sample.duration if present; otherwise infer from raw buffer.
                total_ms = float(getattr(chosen_sample, "duration", 0.0)) * 1000.0
                if total_ms <= 0.0:
                    total_ms = 1000.0 * (len(chosen_sample.data) / max(1, chosen_sample.sr))

                a_ms, h_ms, d_ms = random_ahd_envelope_ms(
                    rng=track.rng,
                    total_ms=total_ms,
                    attack_ms_min=self.env_attack_ms_min,
                    attack_ms_max=self.env_attack_ms_max,
                    hold_ms_min=self.env_hold_ms_min,
                    hold_ms_max=self.env_hold_ms_max,
                    decay_ms_min=self.env_decay_ms_min,
                    decay_ms_max=self.env_decay_ms_max,
                )
            else:
                a_ms = h_ms = d_ms = 0.0

            eff_step = Step(
                index=step.index,
                sample_idx=step.sample_idx,
                on=step.on,
                prob=step.prob,
                semitone=step.semitone,
                gain=step.gain * intensity_mult,
                lowpass=step.lowpass,
                attack_ms=a_ms,
                hold_ms=h_ms,
                decay_ms=d_ms,
            )
            results.append((track, chosen_sample, used_stack, reversed_play, eff_step, intensity_mult))

        return results

    def commit_played_sample_to_stack(self, track: Track, chosen_sample: Sample, used_stack: bool):
        if not self.stack_enabled or used_stack:
            return
        try:
            idx = track.samples.index(chosen_sample)
        except ValueError:
            return
        with track.stack_lock:
            if idx not in track.played_stack:
                track.played_stack.append(idx)
                track.stack_ages[idx] = 0

            while len(track.played_stack) > self.stack_max:
                old_idx = track.played_stack.popleft()
                track.stack_ages.pop(old_idx, None)

    def age_and_decay_stacks(self):
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

    def record_event(
        self,
        step_sample_pos: int,
        global_step: int,
        step_index: int,
        track: Track,
        sample: Sample,
        used_stack: bool,
        reversed_play: bool,
        step_obj: Step,
        intensity_mult: float,
    ):
        ev = {
            "time": time.time(),
            "sample_pos": int(step_sample_pos),
            "global_step": int(global_step),
            "step_index": int(step_index),
            "track_id": int(track.id),
            "track_type": track.stem_type,
            "channel": int(track.channel),
            "sample_name": sample.name,
            "gain": float(step_obj.gain),
            "semitone": float(step_obj.semitone),
            "lowpass": float(step_obj.lowpass),
            "prob": float(step_obj.prob),
            "used_stack": bool(used_stack),
            "reversed": bool(reversed_play),
            "env": {
                "type": "AHD",
                "attack_ms": float(step_obj.attack_ms),
                "hold_ms": float(step_obj.hold_ms),
                "decay_ms": float(step_obj.decay_ms),
            },
            "bpm": float(self.bpm),
            "steps": int(self.steps),
            "repeat_min": int(self.repeat_min),
            "repeat_max": int(self.repeat_max),
            "density": float(self.density),
            "intensity_base": float(self.intensity_base),
            "intensity_step_mult": float(intensity_mult),
        }
        self.run_logger.log_event(ev)


# ---------------- Audio Engine -----------------

class AudioEngine:
    def __init__(self, sequencer: Sequencer, cfg: Dict[str, Any], rng: random.Random):
        self.seq = sequencer
        self.cfg = cfg
        self.rng = rng

        self.sr = int(cfg_get(cfg, "audio.sr"))
        self.blocksize = int(cfg_get(cfg, "audio.blocksize"))
        self.channels_out = int(cfg_get(cfg, "audio.channels"))
        self.latency = cfg_get(cfg, "audio.latency")

        self.fade_samples = int(float(cfg_get(cfg, "processing.fade_duration")) * self.sr)

        self.env_enabled = bool(cfg_get(cfg, "processing.envelope.enabled", True))

        self.clip_mode = str(cfg_get(cfg, "clip.mode", "tube"))
        self.clip_drive = float(cfg_get(cfg, "clip.drive"))
        self.clip_asym = float(cfg_get(cfg, "clip.asym"))

        self.limiter_enabled = bool(cfg_get(cfg, "limiter.enabled", True))
        if self.limiter_enabled:
            self.limiter = MasterLimiter(
                sr=self.sr,
                threshold=float(cfg_get(cfg, "limiter.threshold", 0.98)),
                attack_ms=float(cfg_get(cfg, "limiter.attack_ms", 2.0)),
                release_ms=float(cfg_get(cfg, "limiter.release_ms", 80.0)),
                pre_gain=float(cfg_get(cfg, "limiter.pre_gain", 1.0)),
            )
        else:
            self.limiter = None

        self.clock = Clock(self.seq.bpm, self.sr, steps_per_beat=4)

        self.active_events: List[PlayEvent] = []
        self.events_lock = threading.Lock()

        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.channels_out,
            dtype="float32",
            latency=self.latency,
            callback=self.callback,
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

    def schedule_play(self, sample: Sample, channel: int, reversed_play: bool, step: Step):
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

        # Apply AHD envelope per-trigger (uses step.attack_ms/hold_ms/decay_ms)
        if self.env_enabled and (step.attack_ms > 0.0 or step.hold_ms > 0.0 or step.decay_ms > 0.0):
            # Clamp envelope to *actual* buffer length post-processing
            total_ms_actual = 1000.0 * (len(sig) / max(1, self.sr))
            a = min(step.attack_ms, total_ms_actual)
            h = min(step.hold_ms, max(0.0, total_ms_actual - a))
            d = min(step.decay_ms, max(0.0, total_ms_actual - a - h))
            sig = apply_ahd_envelope(sig, self.sr, a, h, d)

        # Keep a small safety fade to avoid clicks in edge cases (very short A/D)
        sig = apply_fade(sig, self.fade_samples)

        ev = PlayEvent(np.asarray(sig, dtype=np.float32), channel)
        with self.events_lock:
            self.active_events.append(ev)

    def _mix_active_into(self, out: np.ndarray, start: int, end: int):
        length = end - start
        with self.events_lock:
            for ev in list(self.active_events):
                remaining = ev.buffer.shape[0] - ev.pos
                if remaining <= 0:
                    try:
                        self.active_events.remove(ev)
                    except ValueError:
                        pass
                    continue
                n = min(length, remaining)
                if ev.channel < out.shape[1]:
                    out[start : start + n, ev.channel] += ev.buffer[ev.pos : ev.pos + n]
                ev.pos += n

    def callback(self, outdata, frames, time_info, status):
        out = np.zeros((frames, self.channels_out), dtype=np.float32)

        block_start_sample = self.clock.sample_pos
        step_events = self.clock.advance(frames)
        block_start_sample = self.clock.sample_pos - frames

        local_cursor = 0
        for (abs_pos, global_step) in step_events:
            block_offset = int(abs_pos - block_start_sample)
            if block_offset > local_cursor and block_offset <= frames:
                self._mix_active_into(out, local_cursor, min(block_offset, frames))
                local_cursor = block_offset

            step_index = int(global_step % self.seq.steps)

            decisions = self.seq.decide_for_step(global_step, step_index)
            for (track, sample, used_stack, reversed_play, step_obj, intensity_mult) in decisions:
                self.seq.record_event(abs_pos, global_step, step_index, track, sample, used_stack, reversed_play, step_obj, intensity_mult)
                self.schedule_play(sample, track.channel, reversed_play, step_obj)
                self.seq.commit_played_sample_to_stack(track, sample, used_stack)

            self.seq.age_and_decay_stacks()

            if step_index == (self.seq.steps - 1):
                self.seq.loop_count += 1
                if self.seq.loop_count >= self.seq.loop_target:
                    self.seq.create_pattern_block(reason="regenerate", drift=True)
                    self.clock.set_bpm(self.seq.bpm, reset_phase=False)

        if local_cursor < frames:
            self._mix_active_into(out, local_cursor, frames)

        apply_soft_clip(out, self.clip_mode, self.clip_drive, self.clip_asym)
        if self.limiter is not None:
            self.limiter.process(out)

        outdata[:] = out


# ---------------- Main -----------------

def main():
    cfg = load_config("config.yaml")

    seed = cfg_get(cfg, "random.seed", None)
    seed = int(seed) if seed is not None else None

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    global_rng = random.Random(seed)

    db_path = cfg_get(cfg, "db_path")
    samples_dir = cfg_get(cfg, "samples_dir")
    samples_per_track = int(cfg_get(cfg, "sequencer.samples_per_track"))

    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' not found!")
        sys.exit(1)
    if not os.path.exists(samples_dir):
        print(f"Error: Samples directory '{samples_dir}' not found!")
        sys.exit(1)

    print("Loading samples (deterministic)...")
    samples_by_type = load_samples_from_db(db_path, samples_dir, samples_per_track, global_rng)
    total_samples = sum(len(v) for v in samples_by_type.values())
    if total_samples == 0:
        print("Error: No samples found in database!")
        sys.exit(1)
    print(f"Loaded {total_samples} samples total.")

    run_meta = {
        "seed": seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": cfg,
    }
    run_logger = RunLogger(run_meta)

    sequencer = Sequencer(samples_by_type, cfg, global_rng, run_logger)
    engine = AudioEngine(sequencer, cfg, global_rng)

    try:
        engine.start()
        print("Sequencer running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping...")
        engine.stop()
        out_dir = run_logger.finalize()
        print(f"Run logs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
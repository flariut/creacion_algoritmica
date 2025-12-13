import os
import sys
import math
import random
import threading
import argparse
import sqlite3
import json
from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np
import soundfile as sf
import sounddevice as sd
from collections import deque

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
    
    # Soft clipping
    parser.add_argument('--clip-drive', type=float, default=1.0, help='Soft-clip drive (higher = harder saturation)')
    parser.add_argument('--clip-asym', type=float, default=-0.3, help='Asymmetry -1.0..1.0 where positive biases positive side (tube-like)')
    parser.add_argument('--clip-makeup', type=float, default=3, help='Makeup gain applied prior to soft clip')

    # Random control
    parser.add_argument('--seed', type=int, default=None,help='Random seed for reproducibility (default: random)')

    return parser.parse_args()

# ---------------- Channel Mapping ----------------
def get_channel_mapping(output_channels: int) -> Dict[str, int]:
    """Map stem types to output channels based on channel configuration"""
    if output_channels == 1:
        # Mono: all stems go to channel 0
        return {'drums': 0, 'bass': 0, 'other': 0, 'vocals': 0}
    elif output_channels == 2:
        # Stereo: drums/bass left (0), other/vocals right (1)
        return {'drums': 0, 'bass': 0, 'other': 1, 'vocals': 1}
    else:  # 4 channels
        # Quadraphonic: each stem gets its own channel
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
    if len(signal) <= fade_samples * 2:
        # For very short signals, just do a quick fade
        fade_len = min(fade_samples, len(signal) // 2)
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
        # Normal fade in/out
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
    """Apply an ADSR envelope over the length of the signal.
       Envelope times are in seconds. Release is tacked on to the end if short,
       but to avoid length changes we place release within the sample tail if necessary.
    """
    length = signal.shape[0]
    # convert to samples
    a = max(0, int(round(attack * sr)))
    d = max(0, int(round(decay * sr)))
    r = max(0, int(round(release * sr)))
    # sustain region length = remainder
    sustain_len = max(0, length - (a + d + r))
    # if signal too short, scale segments proportionally
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
        env[pos:pos+a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    pos += a
    if d > 0:
        env[pos:pos+d] = np.linspace(1.0, sustain, d, dtype=np.float32)
    pos += d
    # sustain region already set to sustain
    pos += sustain_len
    if r > 0:
        # release falling from sustain to 0
        env[pos:pos+r] = np.linspace(sustain, 0.0, r, dtype=np.float32)
    return signal * env

def soft_clip_channel(x: np.ndarray, drive: float = 1.0, asym: float = 0.0) -> np.ndarray:
    """Soft clip a 1D signal vector.
       drive: pre-saturation gain.
       asym: -1..1 where positive amplifies positive half-wave relative to negative (tube-like).
    """
    # apply pre-drive
    y = x * drive

    # asymmetry - implement different scaling for pos/neg before tanh
    # map asym from -1..1 to pos_scale and neg_scale multipliers
    # keep overall energy normalized: pos_scale + neg_scale ~ 2
    pos_scale = 1.0 + max(0.0, asym)
    neg_scale = 1.0 + max(0.0, -asym)

    # process halves
    pos = np.tanh(pos_scale * np.maximum(0.0, y))
    neg = np.tanh(neg_scale * np.minimum(0.0, y))
    return pos + neg

def apply_soft_clip(out: np.ndarray, drive: float, asym: float, makeup: float):
    """Apply per-channel soft clipping with optional makeup gain."""
    # apply makeup gain first (simple)
    if makeup != 1.0:
        out *= makeup
    # process each channel separately
    for ch in range(out.shape[1]):
        out[:, ch] = soft_clip_channel(out[:, ch], drive, asym)
    # final safety clamp to avoid NaNs/outliers
    np.clip(out, -1.0, 1.0, out)
    return out

# ---------------- Sample Database Loader -----------------
@dataclass
class Sample:
    data: np.ndarray
    sr: int
    name: str
    stem_type: str
    original_position: float
    rms: float

class SampleDatabase:
    def __init__(self, db_path: str, samples_dir: str, rng: random.Random):
        self.db_path = db_path
        self.samples_dir = samples_dir
        self.conn = sqlite3.connect(db_path)
        self.rng = rng  # deterministic RNG passed from main
    
    def get_samples_by_stem_type(self, stem_type: str, limit: int = 100) -> List[Sample]:
        """Fetch samples of a given stem type with deterministic ordering."""
        cursor = self.conn.cursor()
        
        # Deterministic ordering — DO NOT use RANDOM().
        query = """
            SELECT sample_filename, stem_type, sample_position, duration, metadata_json
            FROM samples
            WHERE stem_type = ?
            ORDER BY sample_filename ASC
        """
        
        cursor.execute(query, (stem_type,))
        rows = cursor.fetchall()

        # Deterministic shuffle (instead of SQLite RANDOM)
        self.rng.shuffle(rows)
        if limit is not None:
            rows = rows[:limit]

        samples = []
        for row in rows:
            filename, stem_type, position, duration, metadata_json = row
            
            try:
                audio_path = os.path.join(self.samples_dir, filename)
                if os.path.exists(audio_path):
                    data, sr = sf.read(audio_path, always_2d=False)
                    data = np.asarray(data, dtype=np.float32)
                    
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    sample = Sample(
                        data=data,
                        sr=sr,
                        name=f"{metadata.get('artist', 'Unknown')} - "
                             f"{metadata.get('song_title', 'Unknown')} - {stem_type}",
                        stem_type=stem_type,
                        original_position=position,
                        rms=metadata.get('rms', 0.5)
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
    """Load samples for each stem type from database"""
    db = SampleDatabase(db_path, samples_dir, rng)
    
    stem_types = ['drums', 'bass', 'other', 'vocals']
    samples_by_type = {}
    
    for stem_type in stem_types:
        samples = db.get_samples_by_stem_type(stem_type, samples_per_track)
        samples_by_type[stem_type] = samples
        
        if not samples:
            print(f"Warning: No {stem_type} samples found in database!")
    
    db.close()
    return samples_by_type

# ---------------- Sequencer data ----------------
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
    def random(cls, steps=16, sample_count=1, rng: random.Random = None, pattern_density=0.5, pattern_intensity=1.0):
        rng = rng or random
        p = cls()
        for i in range(steps):
            p.steps.append(Step(
                index=i,
                sample_idx=rng.randrange(max(1, sample_count)),
                on=rng.random() < pattern_density,
                prob=rng.uniform(0.1, 1),
                #semitone=rng.uniform(-12, 12),
                semitone=0,
                gain=rng.uniform(0.1 * pattern_intensity, 0.2 * pattern_intensity),
                #lowpass=rng.uniform(1000.0, 20000.0)
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
    played_stack: deque = field(default_factory=lambda: deque(maxlen=64))
    stack_lock: threading.Lock = field(default_factory=threading.Lock)

@dataclass
class PlayEvent:
    buffer: np.ndarray
    channel: int
    pos: int = 0

# ---------------- Audio engine with callback ----------------
class SampleAccurateSequencer:
    def __init__(self, samples_by_type: Dict[str, List[Sample]], config, rng: random.Random):
        self.samples_by_type = samples_by_type
        self.config = config

        # Sample global rng
        self.global_rng = rng
        
        self.bpm = config.bpm
        self.steps = config.steps
        self.sr = config.sr
        self.blocksize = config.blocksize
        self.channels_out = config.channels
        self.latency = config.latency
        self.fade_samples = int(config.fade_duration * config.sr)

        # envelope params
        self.env_attack = config.env_attack
        self.env_decay = config.env_decay
        self.env_sustain = config.env_sustain
        self.env_release = config.env_release

        # soft clip params
        self.clip_drive = config.clip_drive
        self.clip_asym = config.clip_asym
        self.clip_makeup = config.clip_makeup

        # stack params
        self.stack_enabled = bool(config.stack)
        self.stack_prob = config.stack_prob
        self.stack_max = max(1, config.stack_max)

        # reverse prob
        self.reverse_prob = max(0.0, min(1.0, config.reverse_prob))

        # Get channel mapping
        self.channel_map = get_channel_mapping(self.channels_out)

        # compute step length in samples (16th notes)
        quarter_note_samples = int(round(self.sr * 60.0 / self.bpm))
        self.step_length = int(round(quarter_note_samples / 4.0))

        # callback state
        self.sample_pos = 0
        self.next_step_sample = self.step_length
        self.step_index = 0
        self.loop_count = 0
        self.loop_target = self.global_rng.randint(config.repeat_min, config.repeat_max)

        # active sounds
        self.active_events: List[PlayEvent] = []
        self.lock = threading.Lock()

        # build tracks with patterns and per-track RNG
        self.tracks: List[Track] = []
        
        initial_density = self.global_rng.random()
        print(f"Initial density: {initial_density:.2f}")
        initial_intensity = self.global_rng.uniform(1.0, 5.0)
        print(f"Initial intensity: {initial_intensity:.2f}")

        stem_types = ['drums', 'bass', 'other', 'vocals']
        for i, stem_type in enumerate(stem_types):
            samples = samples_by_type.get(stem_type, [])
            track_rng = random.Random(self.global_rng.randint(0, 2**30))
            pat = Pattern.random(self.steps, max(1, len(samples)), rng=track_rng, pattern_density=initial_density, pattern_intensity=initial_intensity)
            
            track = Track(
                id=i,
                stem_type=stem_type,
                channel=self.channel_map[stem_type],
                samples=samples,
                pattern=pat,
                rng=track_rng
            )
            track.played_stack = deque(maxlen=self.stack_max)
            self.tracks.append(track)
            print(f"Track {i}: {stem_type} -> channel {track.channel}, {len(samples)} samples")

        # sounddevice stream
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            blocksize=self.blocksize,
            channels=self.channels_out,
            dtype='float32',
            latency=self.latency,
            callback=self.callback
        )

        self.running = False

    def start(self):
        self.stream.start()
        self.running = True
        print(f"\nSequencer started:")
        print(f"  BPM: {self.bpm}, Steps: {self.steps}")
        print(f"  Sample rate: {self.sr}, Channels: {self.channels_out}")
        print(f"  Step length: {self.step_length} samples")
        print(f"  Fade: {self.config.fade_duration*1000:.1f}ms ({self.fade_samples} samples)")
        print(f"  Pattern repetitions: {self.loop_target}\n")

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.running = False

    def callback(self, outdata, frames, time_info, status):
        """Audio callback with sample-accurate timing"""
        # create output block
        out = np.zeros((frames, self.channels_out), dtype=np.float32)

        # current buffer absolute sample indices
        start_pos = self.sample_pos
        end_pos = self.sample_pos + frames

        # handle step boundaries within this block
        local_cursor = 0
        while self.next_step_sample < end_pos:
            step_offset = self.next_step_sample - start_pos
            if step_offset > local_cursor:
                self._mix_active_into(out, local_cursor, step_offset)
                local_cursor = step_offset

            # Trigger step at exact sample position
            self._trigger_step(self.step_index)
            
            # Advance to next step
            self.step_index = (self.step_index + 1) % self.steps
            if self.step_index == 0:
                self.loop_count += 1
                if self.loop_count >= self.loop_target:
                    self._regenerate_patterns()
                    self.loop_count = 0
            self.next_step_sample += self.step_length

        # mix remaining active events
        if local_cursor < frames:
            self._mix_active_into(out, local_cursor, frames)

        # advance sample_pos
        self.sample_pos += frames

        # cleanup finished events
        with self.lock:
            self.active_events = [ev for ev in self.active_events if ev.pos < ev.buffer.shape[0]]

        # soft clip and output (per-channel)
        apply_soft_clip(out, self.clip_drive, self.clip_asym, self.clip_makeup)
        outdata[:] = out

    def _mix_active_into(self, out: np.ndarray, start: int, end: int):
        """Mix active events into output buffer"""
        length = end - start
        with self.lock:
            for ev in self.active_events:
                remaining = ev.buffer.shape[0] - ev.pos
                if remaining <= 0:
                    continue
                n = min(length, remaining)
                # bounds safety for channel index
                if ev.channel < out.shape[1]:
                    out[start:start + n, ev.channel] += ev.buffer[ev.pos:ev.pos + n]
                ev.pos += n

    def _trigger_step(self, step_index: int):
        """Trigger samples for all tracks at current step"""
        for track in self.tracks:
            step = track.pattern.steps[step_index]
            if step.maybe_trigger(track.rng) and track.samples:
                # Decide whether to pick from stack (if enabled)
                use_stack_choice = False
                chosen_sample = None
                if self.stack_enabled and (track.rng.random() < self.stack_prob):
                    with track.stack_lock:
                        if len(track.played_stack) > 0:
                            chosen_sample = track.played_stack.pop()
                            use_stack_choice = True
                            print(f"using stack {chosen_sample.name} in track {track.id}")
                if not use_stack_choice:
                    samp = track.samples[step.sample_idx % len(track.samples)]
                    chosen_sample = samp
                    print(f"playing {chosen_sample.name} in track {track.id}")

                # Process sample
                sig = chosen_sample.data.copy()

                # chance to reverse
                if track.rng.random() < self.reverse_prob:
                    sig = sig[::-1]

                # Resample if needed
                if chosen_sample.sr != self.sr:
                    res_ratio = chosen_sample.sr / self.sr
                    sig = resample_linear(sig, res_ratio)
                
                # Apply pitch shift
                pitch_ratio = semitone_to_ratio(step.semitone)
                if pitch_ratio != 1.0:
                    sig = resample_linear(sig, pitch_ratio)
                
                # Apply gain
                sig = apply_gain(sig, step.gain)
                
                # Apply lowpass filter
                sig = lowpass_4pole(sig, step.lowpass, self.sr)
                
                # Ensure mono for consistent processing
                if sig.ndim > 1:
                    sig = np.mean(sig, axis=1)
                
                # Apply ADSR envelope
                sig = apply_adsr_envelope(sig, self.sr, self.env_attack, self.env_decay, self.env_sustain, self.env_release)
                
                # Apply fade in/out to prevent clicks (keeps your original protection)
                sig = apply_fade(sig, self.fade_samples, self.sr)
                
                # Create play event
                with self.lock:
                    self.active_events.append(
                        PlayEvent(np.asarray(sig, dtype=np.float32), track.channel)
                    )

                # push into stack
                if self.stack_enabled and not use_stack_choice:
                    with track.stack_lock:
                        if all(chosen_sample is not s for s in track.played_stack):
                            track.played_stack.append(chosen_sample)
                            print(f"storing {chosen_sample.name} in stack for track {track.id}")

    def _regenerate_patterns(self):
        """Regenerate patterns for all tracks"""
        density = self.global_rng.random()
        print(f"New density: {density:.2f}")
        intensity = self.global_rng.uniform(1.0, 5.0)
        print(f"New intensity: {intensity:.2f}")
        for track in self.tracks:
            track.pattern = Pattern.random(self.steps, max(1, len(track.samples)), rng=track.rng, pattern_density=density, pattern_intensity=intensity)
        
        self.loop_target = self.global_rng.randint(self.config.repeat_min, self.config.repeat_max)
        print(f"[Regenerated patterns. Next repeat: {self.loop_target}]")

# ---------------- Main entry --------------------
def main():
    args = parse_arguments()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"Error: Database file '{args.db_path}' not found!")
        print("Please run the microsample database builder first.")
        sys.exit(1)
    
    if args.seed:
        print(f"Random seed: ")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    global_rng = random.Random(args.seed)

    # Check if samples directory exists
    if not os.path.exists(args.samples_dir):
        print(f"Error: Samples directory '{args.samples_dir}' not found!")
        sys.exit(1)
    
    # Load samples from database
    print("Loading samples from database...")
    samples_by_type = load_samples_from_db(args.db_path, args.samples_dir, args.samples_per_track, global_rng)
    
    # Check if we have enough samples
    total_samples = sum(len(samples) for samples in samples_by_type.values())
    if total_samples == 0:
        print("Error: No samples found in database!")
        sys.exit(1)
    
    print(f"\nLoaded {total_samples} total samples from {args.samples_dir}:")
    for stem_type, samples in samples_by_type.items():
        print(f"  {stem_type}: {len(samples)} samples")
    
    # Create and start sequencer
    sequencer = SampleAccurateSequencer(samples_by_type, args, global_rng)
    
    try:
        sequencer.start()
        print("Sequencer running. Press Ctrl+C to stop.")
        
        while True:
            try:
                threading.Event().wait(1.0)
            except KeyboardInterrupt:
                break
                
    finally:
        print("Stopping sequencer...")
        sequencer.stop()

if __name__ == "__main__":
    main()

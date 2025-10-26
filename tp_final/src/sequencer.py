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
    parser.add_argument('--blocksize', type=int, default=2048, help='Audio block size')
    parser.add_argument('--latency', default='high', help='Audio latency setting')
    
    # Sequence configuration
    parser.add_argument('--repeat-min', type=int, default=4, help='Minimum pattern repetitions')
    parser.add_argument('--repeat-max', type=int, default=16, help='Maximum pattern repetitions')
    parser.add_argument('--samples-per-track', type=int, default=1000, 
                       help='Number of samples to fetch per track type')
    
    # Sample processing
    parser.add_argument('--fade-duration', type=float, default=0.005, 
                       help='Fade in/out duration in seconds (default: 5ms)')
    
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
    def __init__(self, db_path: str, samples_dir: str):
        self.db_path = db_path
        self.samples_dir = samples_dir
        self.conn = sqlite3.connect(db_path)
        
    def get_samples_by_stem_type(self, stem_type: str, limit: int = 100) -> List[Sample]:
        """Fetch samples of specific stem type from database"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT sample_filename, stem_type, sample_position, duration, metadata_json
        FROM samples 
        WHERE stem_type = ?
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        cursor.execute(query, (stem_type, limit))
        rows = cursor.fetchall()
        
        samples = []
        for row in rows:
            filename, stem_type, position, duration, metadata_json = row
            
            # Load audio file from samples directory
            try:
                audio_path = os.path.join(self.samples_dir, filename)
                if os.path.exists(audio_path):
                    data, sr = sf.read(audio_path, always_2d=False)
                    data = np.asarray(data, dtype=np.float32)
                    
                    # Parse metadata
                    metadata = json.loads(metadata_json) if metadata_json else {}
                    
                    sample = Sample(
                        data=data,
                        sr=sr,  # Get sample rate from the audio file itself
                        name=f"{metadata.get('artist', 'Unknown')} - {stem_type}",
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

def load_samples_from_db(db_path: str, samples_dir: str, samples_per_track: int) -> Dict[str, List[Sample]]:
    """Load samples for each stem type from database"""
    db = SampleDatabase(db_path, samples_dir)
    
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
    def random(cls, steps=16, sample_count=1, rng: random.Random = None):
        rng = rng or random
        p = cls()
        for i in range(steps):
            p.steps.append(Step(
                index=i,
                sample_idx=rng.randrange(max(1, sample_count)),
                on=rng.choice([True, False]),
                prob=rng.uniform(0.1, 1),
                #prob=1,
                #semitone=rng.uniform(-12, 12),
                semitone=0,
                gain=rng.uniform(0.5, 1.0),
                #lowpass=rng.uniform(1000.0, 12000.0)
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

@dataclass
class PlayEvent:
    buffer: np.ndarray
    channel: int
    pos: int = 0

# ---------------- Audio engine with callback ----------------
class SampleAccurateSequencer:
    def __init__(self, samples_by_type: Dict[str, List[Sample]], config):
        self.samples_by_type = samples_by_type
        self.config = config
        
        self.bpm = config.bpm
        self.steps = config.steps
        self.sr = config.sr
        self.blocksize = config.blocksize
        self.channels_out = config.channels
        self.latency = config.latency
        self.fade_samples = int(config.fade_duration * config.sr)

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
        self.loop_target = random.randint(config.repeat_min, config.repeat_max)

        # active sounds
        self.active_events: List[PlayEvent] = []
        self.lock = threading.Lock()

        # build tracks with patterns and per-track RNG
        self.tracks: List[Track] = []
        global_rng = random.Random()
        
        stem_types = ['drums', 'bass', 'other', 'vocals']
        for i, stem_type in enumerate(stem_types):
            samples = samples_by_type.get(stem_type, [])
            track_rng = random.Random(global_rng.randint(0, 2**30))
            pat = Pattern.random(self.steps, max(1, len(samples)), rng=track_rng)
            
            track = Track(
                id=i,
                stem_type=stem_type,
                channel=self.channel_map[stem_type],
                samples=samples,
                pattern=pat,
                rng=track_rng
            )
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

        # clip and output
        np.clip(out, -1.0, 1.0, out)
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
                out[start:start + n, ev.channel] += ev.buffer[ev.pos:ev.pos + n]
                ev.pos += n

    def _trigger_step(self, step_index: int):
        """Trigger samples for all tracks at current step"""
        for track in self.tracks:
            step = track.pattern.steps[step_index]
            if step.maybe_trigger(track.rng) and track.samples:
                # Get sample
                samp = track.samples[step.sample_idx % len(track.samples)]
                
                # Process sample
                sig = samp.data
                
                # Resample if needed
                if samp.sr != self.sr:
                    res_ratio = samp.sr / self.sr
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
                
                # Apply fade in/out to prevent clicks
                sig = apply_fade(sig, self.fade_samples, self.sr)
                
                # Create play event
                with self.lock:
                    self.active_events.append(
                        PlayEvent(np.asarray(sig, dtype=np.float32), track.channel)
                    )

    def _regenerate_patterns(self):
        """Regenerate patterns for all tracks"""
        for track in self.tracks:
            track.pattern = Pattern.random(self.steps, max(1, len(track.samples)), rng=track.rng)
        
        self.loop_target = random.randint(self.config.repeat_min, self.config.repeat_max)
        print(f"[Regenerated patterns. Next repeat: {self.loop_target}]")

# ---------------- Main entry --------------------
def main():
    args = parse_arguments()
    
    # Check if database exists
    if not os.path.exists(args.db_path):
        print(f"Error: Database file '{args.db_path}' not found!")
        print("Please run the microsample database builder first.")
        sys.exit(1)
    
    # Check if samples directory exists
    if not os.path.exists(args.samples_dir):
        print(f"Error: Samples directory '{args.samples_dir}' not found!")
        sys.exit(1)
    
    # Load samples from database
    print("Loading samples from database...")
    samples_by_type = load_samples_from_db(args.db_path, args.samples_dir, args.samples_per_track)
    
    # Check if we have enough samples
    total_samples = sum(len(samples) for samples in samples_by_type.values())
    if total_samples == 0:
        print("Error: No samples found in database!")
        sys.exit(1)
    
    print(f"\nLoaded {total_samples} total samples from {args.samples_dir}:")
    for stem_type, samples in samples_by_type.items():
        print(f"  {stem_type}: {len(samples)} samples")
    
    # Create and start sequencer
    sequencer = SampleAccurateSequencer(samples_by_type, args)
    
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
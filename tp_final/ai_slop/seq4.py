import os
import sys
import math
import random
import threading
from dataclasses import dataclass, field
from typing import List
import numpy as np
import soundfile as sf
import sounddevice as sd

# ---------------- Configuration ----------------
SOUNDS_DIR = './sounds'
BPM = 132
STEPS = 16
TRACKS = 2
REPEAT_MIN, REPEAT_MAX = 1, 1
SR = 44100
BLOCKSIZE = 128
LATENCY = 'high'
TRACK_TO_CHANNEL = list(range(TRACKS))

# ---------------- DSP helpers -------------------
def semitone_to_ratio(semitones: float) -> float:
    return 2.0 ** (semitones / 12.0)

def resample_linear(signal: np.ndarray, ratio: float) -> np.ndarray:
    """
    Fast linear resampling: new_length = floor(len(signal) / ratio)
    ratio > 1 -> speed up (shorter), ratio < 1 -> slow down (longer).
    Works for mono (1D) or stereo/multi (2D).
    """
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
    """
    Very simple cascade of 4 one-pole filters (cheap emulation).
    Accepts mono or stereo (will convert to mono internally).
    """
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

# ---------------- Sample loader -----------------
@dataclass
class Sample:
    data: np.ndarray  # 1D or 2D
    sr: int
    name: str

def load_samples(folder: str) -> List[Sample]:
    lst = []
    if not os.path.isdir(folder):
        print(f"Sounds folder '{folder}' not found.", file=sys.stderr)
        return lst
    for fn in sorted(os.listdir(folder)):
        path = os.path.join(folder, fn)
        if os.path.isfile(path) and fn.lower().endswith(('.wav', '.flac', '.aiff', '.aif')):
            try:
                data, sr = sf.read(path, always_2d=False)
                data = np.asarray(data, dtype=np.float32)
                lst.append(Sample(data, sr, fn))
                print(f"Loaded {fn} ({data.shape}, {sr} Hz)")
            except Exception as e:
                print(f"Failed to load {fn}: {e}", file=sys.stderr)
    if not lst:
        print("No samples found in ./sounds", file=sys.stderr)
    return lst

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
                prob=rng.uniform(0.4, 0.95),
                semitone=rng.uniform(-12, 12),
                gain=rng.uniform(0.5, 1.0),
                lowpass=rng.uniform(1000.0, 12000.0)
            ))
        return p

@dataclass
class Track:
    id: int
    channel: int
    pattern: Pattern
    rng: random.Random  # per-track RNG for deterministic randomness if desired

@dataclass
class PlayEvent:
    buffer: np.ndarray  # 1D mono float32
    channel: int
    pos: int = 0

# ------------- Motif / Variation Generator ---------------
def mutate_step(step: Step, rng: random.Random, mutation_prob=0.25) -> Step:
    # make a shallow copy and apply small mutations
    s = Step(
        index=step.index,
        sample_idx=step.sample_idx,
        on=step.on,
        prob=step.prob,
        semitone=step.semitone,
        gain=step.gain,
        lowpass=step.lowpass
    )
    # toggle on/off occasionally
    if rng.random() < mutation_prob:
        s.on = not s.on
    # nudge probability
    if rng.random() < mutation_prob:
        s.prob = max(0.05, min(1.0, s.prob + rng.uniform(-0.25, 0.25)))
    # small pitch variation (a few semitones)
    if rng.random() < mutation_prob:
        s.semitone = max(-24, min(24, s.semitone + rng.uniform(-3, 3)))
    # small gain variation
    if rng.random() < mutation_prob:
        s.gain = max(0.1, min(2.0, s.gain * (1.0 + rng.uniform(-0.3, 0.3))))
    # lowpass tweak
    if rng.random() < mutation_prob:
        s.lowpass = max(200.0, min(20000.0, s.lowpass + rng.uniform(-2000.0, 2000.0)))
    return s

def vary_pattern_block(base_block: List[Step], rng: random.Random, shift_range=1, mutation_prob=0.25):
    # apply per-step small mutations and maybe roll (shift) the block
    new_block = [mutate_step(st, rng, mutation_prob) for st in base_block]
    if shift_range != 0:
        shift = rng.randint(-shift_range, shift_range)
        if shift != 0:
            new_block = new_block[-shift:] + new_block[:-shift]
    # reindex
    for i, st in enumerate(new_block):
        st.index = i
    return new_block

def generate_section(tracks_n: int, samples_count: int, steps_total: int, rngs: List[random.Random]):
    """
    Build patterns per track for one section based on motifs and variations.
    - We use 3 base motifs (A,B,C)
    - A structure of 8 tokens is chosen (each token covers steps_total/8 steps)
    - Each token is a variation of a base motif (A', A'', etc.)
    Returns: list_of_Patterns_per_track, structure_tokens
    """
    # ensure steps_total divisible by token_count (we use 8 tokens)
    token_count = 8
    if steps_total % token_count != 0:
        raise ValueError("STEPS must be divisible by 8 for this generator (current STEPS=%d)" % steps_total)
    token_len = steps_total // token_count  # e.g., 16/8 = 2 steps per token

    # pick structure template
    structures = [
        ["A","A'","B","B'","A''","A'''","B''","B'''"],
        ["A","A'","B","C","A''","A'''","B'","C'"],
        ["A","B","A'","B'","A''","B''","C","C'"],
        ["A","A'","A''","B","B'","B''","C","C'"]
    ]
    structure = rngs[0].choice(structures)

    # Build base motifs for A,B,C for each track: each motif is token_len long
    base_motifs = {'A': [], 'B': [], 'C': []}
    for motif_name in base_motifs.keys():
        for tr in range(tracks_n):
            rng = rngs[tr]
            block = []
            # Create base block with some density / spacing using simple rules
            # E.g., place hits on certain beats within token_len
            for i in range(token_len):
                sample_idx = rng.randrange(max(1, samples_count))
                # Start with an on/off pattern that is not identical across motifs
                if motif_name == 'A':
                    on = (i % max(1, token_len//1) == 0)  # sparser
                elif motif_name == 'B':
                    on = (i % max(1, token_len//2) == 0)
                else:  # C
                    on = (rng.random() < 0.5)
                step = Step(
                    index=i,
                    sample_idx=sample_idx,
                    on=on,
                    prob=rng.uniform(0.6, 0.95),
                    semitone=rng.uniform(-4, 4),
                    gain=rng.uniform(0.7, 1.0),
                    lowpass=rng.uniform(3000.0, 12000.0)
                )
                block.append(step)
            base_motifs[motif_name].append(block)

    # For each token in structure, create a variation per track and append to track pattern
    patterns_per_track = [Pattern() for _ in range(tracks_n)]
    for token in structure:
        base_name = token[0]  # 'A' from "A''"
        for tr in range(tracks_n):
            base_block = base_motifs[base_name][tr]
            var_block = vary_pattern_block(base_block, rngs[tr], shift_range=1, mutation_prob=0.3)
            # append var_block to pattern, adjusting indices to global positions later
            patterns_per_track[tr].steps.extend(var_block)

    # Now reindex steps to 0..steps_total-1
    for tr in range(tracks_n):
        pat = patterns_per_track[tr]
        # If by any chance length mismatches, trim or extend with random steps
        if len(pat.steps) > steps_total:
            pat.steps = pat.steps[:steps_total]
        elif len(pat.steps) < steps_total:
            # fill remaining with mutated copies of last block
            need = steps_total - len(pat.steps)
            for i in range(need):
                # duplicate last step with slight mutation
                last = pat.steps[-1] if pat.steps else Step(0, 0, True, 0.9, 0.0, 1.0, 15000.0)
                pat.steps.append(mutate_step(last, rngs[tr], mutation_prob=0.4))
        for idx, st in enumerate(pat.steps):
            st.index = idx

    return patterns_per_track, structure

# ---------------- Audio engine with callback ----------------
class SampleAccurateSequencer:
    def __init__(self, samples: List[Sample], bpm=BPM, steps=STEPS, tracks=TRACKS,
                 sr=SR, blocksize=BLOCKSIZE, channels_out=TRACKS, latency=LATENCY):
        self.samples = samples
        self.bpm = bpm
        self.steps = steps
        self.tracks_n = tracks
        self.sr = sr
        self.blocksize = blocksize
        self.channels_out = channels_out
        self.latency = latency

        # compute step length in samples (16th notes)
        quarter_note_samples = int(round(self.sr * 60.0 / self.bpm))
        self.step_length = int(round(quarter_note_samples / 4.0))  # 16th note in samples

        # callback state
        self.sample_pos = 0  # absolute sample index since start
        self.next_step_sample = self.step_length  # sample index where next step boundary occurs
        self.step_index = 0
        self.loop_count = 0
        self.loop_target = random.randint(REPEAT_MIN, REPEAT_MAX)

        # active sounds
        self.active_events: List[PlayEvent] = []
        self.lock = threading.Lock()

        # build tracks with patterns and per-track RNG
        self.tracks: List[Track] = []
        global_rng = random.Random()
        # create per-track RNGs
        rngs = []
        for i in range(self.tracks_n):
            rngs.append(random.Random(global_rng.randint(0, 2**30)))
        # generate initial structured section patterns
        patterns, structure = generate_section(self.tracks_n, max(1, len(self.samples)), self.steps, rngs)
        self.current_structure = structure
        print("Initial structure:", " - ".join(self.current_structure))
        for i in range(self.tracks_n):
            self.tracks.append(Track(i, TRACK_TO_CHANNEL[i % channels_out], patterns[i], rngs[i]))

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

        print("Generated initial patterns. Repeat target:", self.loop_target)

    def start(self):
        self.stream.start()
        self.running = True
        print("Stream started. Step length (samples):", self.step_length,
              "BPM:", self.bpm, "SR:", self.sr)
        print(f"Initial pattern loop target: {self.loop_target} repetitions before regenerate.\n")

    def stop(self):
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
        self.running = False

    def callback(self, outdata, frames, time_info, status):
        """
        This callback is where all triggering and timing decisions happen.
        We advance sample_pos by `frames` per call, and when sample_pos crosses
        next_step_sample we trigger the next step (for all tracks) SAMPLE-ACCURATELY.
        """
        # create output block
        out = np.zeros((frames, self.channels_out), dtype=np.float32)

        # current buffer absolute sample indices: [self.sample_pos, self.sample_pos + frames)
        start_pos = self.sample_pos
        end_pos = self.sample_pos + frames

        # while the next step boundary is within this callback block, we need to trigger at
        # the exact sample offset within the block. There can be 0 or multiple step boundaries per block.
        # We'll iterate handling each boundary in order.
        local_cursor = 0  # offset inside out block where we've processed up to
        while self.next_step_sample < end_pos:
            # compute offset (samples) inside this block where step occurs
            step_offset = self.next_step_sample - start_pos
            # first, mix any currently active events up to step_offset
            if step_offset > local_cursor:
                self._mix_active_into(out, local_cursor, step_offset)
                local_cursor = step_offset

            # Now trigger step at exactly sample index self.next_step_sample
            self._trigger_step(self.step_index)
            # advance step index and compute next_step_sample
            self.step_index = (self.step_index + 1) % self.steps
            if self.step_index == 0:
                # completed a loop of STEPS
                self.loop_count += 1
                if self.loop_count >= self.loop_target:
                    self._regenerate_patterns()
                    self.loop_count = 0
            self.next_step_sample += self.step_length

        # mix any remaining active events for the rest of the block
        if local_cursor < frames:
            self._mix_active_into(out, local_cursor, frames)
            local_cursor = frames

        # advance sample_pos
        self.sample_pos += frames

        # cleanup finished events
        with self.lock:
            self.active_events = [ev for ev in self.active_events if ev.pos < ev.buffer.shape[0]]

        # clip and output
        np.clip(out, -1.0, 1.0, out)
        outdata[:] = out

    def _mix_active_into(self, out: np.ndarray, start: int, end: int):
        """Mix active events into out[start:end, :]"""
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
        """For each track, evaluate its step at sample-accurate time and append PlayEvent if triggered."""
        for track in self.tracks:
            step = track.pattern.steps[step_index]
            if step.maybe_trigger(track.rng):
                # get sample safely
                if not self.samples:
                    continue
                samp = self.samples[step.sample_idx % len(self.samples)]
                # 1) if sample sr != engine sr -> resample to engine SR first (naive)
                sig = samp.data
                if samp.sr != self.sr:
                    res_ratio = samp.sr / self.sr
                    sig = resample_linear(sig, res_ratio)
                # 2) apply pitch by resampling according to semitone ratio
                pitch_ratio = semitone_to_ratio(step.semitone)
                if pitch_ratio != 1.0:
                    sig = resample_linear(sig, pitch_ratio)
                # 3) apply gain
                sig = apply_gain(sig, step.gain)
                # 4) lowpass 4-pole
                sig = lowpass_4pole(sig, step.lowpass, self.sr)
                # ensure mono 1D
                if sig.ndim > 1:
                    sig = np.mean(sig, axis=1)
                # append event (pos=0). Mixing will align using event.pos
                with self.lock:
                    self.active_events.append(PlayEvent(np.asarray(sig, dtype=np.float32), track.channel))

    def _regenerate_patterns(self):
        """Regenerate patterns for all tracks. Called from callback (safe)."""
        # create new RNGs? keep current track RNGs to preserve character
        rngs = [t.rng for t in self.tracks]
        patterns, structure = generate_section(self.tracks_n, max(1, len(self.samples)), self.steps, rngs)
        for i, track in enumerate(self.tracks):
            track.pattern = patterns[i]
        self.current_structure = structure
        self.loop_target = random.randint(REPEAT_MIN, REPEAT_MAX)
        # print info
        print(f"[callback] Regenerated patterns. Structure: {' - '.join(self.current_structure)}. "
              f"Next repeat target: {self.loop_target}")

# ---------------- Main entry --------------------
def main():
    samples = load_samples(SOUNDS_DIR)
    if not samples:
        print("No samples loaded. Put files (wav/flac/aiff) into ./sounds and run again.")
        return

    # detect device channel count (optional, graceful fallback)
    try:
        dev = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
        info = sd.query_devices(dev, 'output')
        max_ch = info['max_output_channels']
        print(f"Using output device: {info['name']} with {max_ch} channels.")
    except Exception:
        max_ch = max(TRACKS, 2)

    channels_out = max(max_ch, TRACKS)

    seq = SampleAccurateSequencer(samples, bpm=BPM, steps=STEPS, tracks=TRACKS,
                                  sr=SR, blocksize=BLOCKSIZE, channels_out=channels_out, latency=LATENCY)
    try:
        seq.start()
        print("Sequencer running. Press Ctrl+C to stop.")
        # main thread just sleeps while callback drives everything
        while True:
            try:
                # small sleep to keep CPU usage low
                threading.Event().wait(1.0)
            except KeyboardInterrupt:
                break
    finally:
        print("Stopping...")
        seq.stop()

if __name__ == "__main__":
    main()

# Default libraries
import argparse
import sqlite3
import shutil
import json
import hashlib
import os
from pathlib import Path

# Custom libs
import librosa
import numpy as np
import soundfile as sf
from audio_separator.separator import Separator as audio_separator
from mutagen import File as mutagen_file

class MicrosampleDatabaseBuilder:
    def __init__(self, config):
        self.config = config
        self.db_path = config['db_path']
        self.samples_output_dir = Path(config['samples_output_dir'])
        self.temp_dir = Path(config.get('temp_dir', 'temp_stems'))
        self.sample_duration = config.get('sample_duration', 0.5)  # seconds
        self.samples_per_stem = config.get('samples_per_stem', 10)
        
        self.min_rms = round(10 ** (config['min_rms_db'] / 20), 2)

        # Create directories
        self.samples_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
            
        # Initialize separation model
        self.separator = audio_separator(output_dir=str(self.temp_dir))
        self.separator.load_model("htdemucs.yaml")

        print(f"Output directory: {self.separator.output_dir}")
        print(f"Model directory: {self.separator.model_file_dir}")
        
        print("Microsample Database Builder initialized successfully")
        print(f"Database: {self.db_path}")
        print(f"Samples output: {self.samples_output_dir}")
        print(f"Temp directory: {self.temp_dir}")
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT NOT NULL,
                artist TEXT,
                album TEXT,
                song_title TEXT,
                original_path TEXT NOT NULL,
                stem_type TEXT NOT NULL,
                sample_position REAL NOT NULL,
                duration REAL NOT NULL,
                sample_filename TEXT NOT NULL,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(file_hash, stem_type, sample_position)
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stem_type ON samples(stem_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_artist ON samples(artist)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_hash ON samples(file_hash)')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully")

    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file for unique identification"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_collection(self, collection_dir):
        """Process entire music collection"""
        audio_files = self.find_audio_files(collection_dir)
        total_files = len(audio_files)
        
        if total_files == 0:
            print(f"No audio files found in {collection_dir}")
            return
        
        print(f"\nStarting processing of {total_files} audio files...")
        
        processed_count = 0
        failed_count = 0
        
        for i, audio_file in enumerate(audio_files):
            print(f"\nProgress: {i+1}/{total_files} ({((i+1)/total_files*100):.1f}%)")
            
            success = self.process_audio_file(audio_file)
            if success:
                processed_count += 1
            else:
                failed_count += 1
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {processed_count}")
        print(f"Failed: {failed_count}")
        print(f"Total: {total_files}")
        print(f"Database: {self.db_path}")
        print(f"Samples directory: {self.samples_output_dir}")


    def find_audio_files(self, root_dir):
        """Recursively find all audio files in directory structure"""
        root_path = Path(root_dir)
        audio_files = []
        
        for ext in ['*.mp3', '*.flac', '*.wav', '*.m4a']:
            audio_files.extend(root_path.rglob(ext))
        
        print(f"Found {len(audio_files)} audio files in {root_dir}")
        return audio_files

    def process_audio_file(self, audio_file):
        """Process a single audio file through the entire pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing: {audio_file}")
        print(f"{'='*60}")
        
        try:
            # Calculate file hash for unique identification
            file_hash = self.calculate_file_hash(audio_file)
            
            # Check if file was already processed
            if self.is_file_processed(file_hash):
                print(f"File already processed: {audio_file}")
                return True
            
            # Extract metadata from path
            metadata = self.extract_metadata_from_path(audio_file)
            metadata['file_hash'] = file_hash
            
            print(f"Metadata: {metadata['artist']} - {metadata['song_title']}")
            
            # Separate stems
            stems = self.separate_audio_stems(audio_file)
            
            if not stems:
                print(f"No stems generated for: {audio_file}")
                return False
            
            total_samples_created = 0
            
            # Process each stem
            for stem_type, stem_data in stems.items():
                audio_data = stem_data['audio']
                sample_rate = stem_data['sample_rate']
                
                print(f"Processing {stem_type} stem...")
                
                # Extract samples from this stem
                samples = self.extract_samples_from_stem(
                    audio_data, stem_type, sample_rate, metadata, file_hash
                )
                
                # Save each sample
                for sample in samples:
                    sample_id = f"{file_hash}_{stem_type}_{sample['sequence']}_{int(sample['position'])}"
                    sample_filename = self.save_sample_to_disk(
                        sample['data'], sample_rate, sample_id
                    )
                    
                    if sample_filename:
                        # Store in database
                        db_id = self.store_sample_in_database(sample, sample_filename)
                        if db_id:
                            total_samples_created += 1
            
            print(f"✓ Created {total_samples_created} samples from {audio_file}")

            for _, stem_data in stems.items():
                try:
                    path = stem_data["file_path"]
                    print(f"Removing stem: {path}")
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing {path}: {e}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    # def save_sample_to_disk(self, sample_data, sample_rate, sample_id):
    #     """Save sample audio to disk"""
    #     filename = f"{sample_id}.wav"
    #     filepath = self.samples_output_dir / filename
        
    #     try:
    #         # Ensure the data is in correct format for soundfile
    #         if len(sample_data.shape) == 1:
    #             # Mono audio
    #             sample_data_to_save = sample_data
    #         elif sample_data.shape[0] == 2 and sample_data.shape[1] > 2:
    #             # Shape: (channels, samples) -> convert to (samples, channels)
    #             sample_data_to_save = sample_data.T
    #         else:
    #             # Assume (samples, channels) or other format
    #             sample_data_to_save = sample_data
            
    #         sf.write(filepath, sample_data_to_save, sample_rate)
    #         return filename
    #     except Exception as e:
    #         print(f"Error saving sample {sample_id}: {e}")
    #         # Fallback: try with mono conversion
    #         try:
    #             if len(sample_data.shape) > 1:
    #                 sample_data = np.mean(sample_data, axis=0)
    #             sf.write(filepath, sample_data, sample_rate)
    #             return filename
    #         except Exception as e2:
    #             print(f"Fallback save also failed: {e2}")
    #             return None

    def save_sample_to_disk(self, sample_data, sample_rate, sample_id):
        """Save sample audio to disk (always mono)"""
        filename = f"{sample_id}.wav"
        filepath = self.samples_output_dir / filename
        
        try:
            # FORCE MONO
            if sample_data.ndim > 1:
                # (channels, samples) → average across channels
                if sample_data.shape[0] < sample_data.shape[1]:
                    sample_data = np.mean(sample_data, axis=0)
                else:
                    sample_data = np.mean(sample_data, axis=1)

            # Now sample_data is guaranteed mono (1D)
            sf.write(filepath, sample_data, sample_rate)
            return filename

        except Exception as e:
            print(f"Error saving sample {sample_id}: {e}")
            return None

    def store_sample_in_database(self, sample_info, sample_filename):
        """Store sample metadata in SQLite database"""
        if sample_filename is None:
            return None
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        metadata_json = json.dumps(sample_info['metadata'])
        
        cursor.execute('''
            INSERT OR IGNORE INTO samples 
            (file_hash, artist, album, song_title, original_path, stem_type, 
             sample_position, duration, sample_filename, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            sample_info['file_hash'],
            sample_info['metadata']['artist'],
            sample_info['metadata']['album'],
            sample_info['metadata']['song_title'],
            sample_info['metadata']['original_path'],
            sample_info['stem_type'],
            sample_info['position'],
            sample_info['duration'],
            sample_filename,
            metadata_json
        ))
        
        sample_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return sample_id
    
    def _normalize_tag_value(self, val):
        """Normalize different mutagen tag value shapes into a plain string (or None)."""
        if val is None:
            return None
        # mutagen often returns lists
        if isinstance(val, (list, tuple)):
            if len(val) == 0:
                return None
            val = val[0]
        # ID3 frames (like TextFrame) expose .text
        if hasattr(val, "text"):
            try:
                return str(val.text[0]) if val.text else None
            except Exception:
                pass
        # MP4 uses bytes or strings inside lists — try bytes decode
        if isinstance(val, bytes):
            try:
                return val.decode('utf-8', errors='replace')
            except Exception:
                return str(val)
        # simple string or numeric
        return str(val)

    def _lookup_tag(self, tags, candidates):
        """Search for any of the candidate tag keys in the tags object and return normalized value."""
        if tags is None:
            return None

        # tags can be dict-like or an ID3 object
        for key in candidates:
            # try direct key
            try:
                v = tags.get(key)
            except Exception:
                v = None

            # some mutagen tag containers let using indexing like tags[key]
            if v is None:
                try:
                    v = tags[key]
                except Exception:
                    v = None

            if v is not None:
                return self._normalize_tag_value(v)

        # last resort: try any key case variation
        try:
            for k in list(tags.keys()):
                kl = k.lower()
                for candidate in candidates:
                    if candidate.lower() == kl:
                        return self._normalize_tag_value(tags.get(k))
        except Exception:
            pass

        return None

    def extract_metadata_from_path(self, file_path):
        """
        Robust metadata extraction for many audio formats (flac, mp3, m4a, wma, ...)
        returns dict with artist, album, song_title, original_path, plus optional genre, track.
        """
        file_path = Path(file_path)
        audio = None
        try:
            audio = mutagen_file(file_path, easy=False)  # easy=False to get full tag objects
        except Exception as e:
            print(f"mutagen failed to open file {file_path}: {e}")
            audio = None

        tags = getattr(audio, "tags", None)

        # Candidate keys for common tag formats
        artist_candidates = ['artist', 'ARTIST', 'ART', 'TPE1', '©ART', 'Author']
        album_candidates  = ['album', 'ALBUM', 'TALB', '©alb']
        title_candidates  = ['title', 'TITLE', 'TIT2', '©nam']
        genre_candidates  = ['genre', 'GENRE', 'TCON', '©gen']
        track_candidates  = ['tracknumber', 'trackNumber', 'TRACKNUMBER', 'TRCK', '©trk']

        # For ID3 (mp3) audio.tags may be a Mutagen ID3 object; its keys are frame names like 'TPE1'
        # For FLAC/Vorbis, keys often are uppercase like 'ARTIST', 'TITLE'
        artist = self._lookup_tag(tags, artist_candidates)
        album = self._lookup_tag(tags, album_candidates)
        song_title = self._lookup_tag(tags, title_candidates)
        genre = self._lookup_tag(tags, genre_candidates)
        track = self._lookup_tag(tags, track_candidates)

        # Some formats (EasyID3/EasyMP4) expose lowercase friendly keys inside audio.tags
        # If still missing, try the "easy" interface fallback:
        if (artist is None or album is None or song_title is None) and audio is not None:
            try:
                easy = mutagen_file(file_path, easy=True)
                easy_tags = getattr(easy, "tags", None)
                if artist is None:
                    artist = self._lookup_tag(easy_tags, ['artist'])
                if album is None:
                    album = self._lookup_tag(easy_tags, ['album'])
                if song_title is None:
                    song_title = self._lookup_tag(easy_tags, ['title'])
                if genre is None:
                    genre = self._lookup_tag(easy_tags, ['genre'])
                if track is None:
                    track = self._lookup_tag(easy_tags, ['tracknumber', 'track'])
            except Exception:
                pass

        # Final fallback unknown
        path_parts = file_path.parts
        if not artist:
            # if len(path_parts) >= 3:
            #     artist = path_parts[-3]
            # elif len(path_parts) == 2:
            #     artist = path_parts[-2]
            # else:
            artist = "Unknown Artist"
        if not album:
            # if len(path_parts) >= 2:
            #     album = path_parts[-2]
            # else:
            album = "Unknown Album"
        if not song_title:
            song_title = file_path.stem

        return {
            'artist': artist,
            'album': album,
            'song_title': song_title,
            'genre': genre,
            'track': track,
            'original_path': str(file_path)
        }

    def separate_audio_stems(self, audio_file):
        """Separate audio file into stems using python-audio-separator"""
        try:
            print(f"Starting stem separation for: {audio_file}")

            output_files = self.separator.separate(str(audio_file))
            
            if not output_files:
                print("No output files generated from separation")
                return {}
            
            print(f"Separation completed. Generated {len(output_files)} stem files:")
            for output_file in output_files:
                print(f"  - {output_file}")
            
            stems = {}
            for output_file in output_files:
                stem_type = self.determine_stem_type(output_file)

                stem_path = self.temp_dir / Path(output_file)

                print(f"Loading {stem_type} stem from: {stem_path}")
                
                try:
                    # Load the separated audio
                    audio_data, sample_rate = librosa.load(stem_path, sr=None, mono=False)
                    stems[stem_type] = {
                        'audio': audio_data,
                        'sample_rate': sample_rate,
                        'file_path': stem_path
                    }
                    print(f"Successfully loaded {stem_type} stem")
                except Exception as e:
                    print(f"Error loading stem {stem_path}: {e}")
            
            return stems
            
        except Exception as e:
            print(f"Error separating stems for {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def determine_stem_type(self, filename):
        """Determine stem type from filename"""
        filename_lower = str(filename).lower()
        
        if 'vocals' in filename_lower:
            return 'vocals'
        elif 'drum' in filename_lower:
            return 'drums'
        elif 'bass' in filename_lower:
            return 'bass'
        elif 'other' in filename_lower:
            return 'other'
        elif 'accompaniment' in filename_lower or 'instrumental' in filename_lower:
            return 'harmony'
        elif 'guitar' in filename_lower:
            return 'guitar'
        elif 'piano' in filename_lower or 'keys' in filename_lower:
            return 'piano'
        else:
            # Try to extract from common patterns
            if any(x in filename_lower for x in ['_vocals', '_vocal', '_sing']):
                return 'vocals'
            elif any(x in filename_lower for x in ['_drums', '_drum', '_perc']):
                return 'drums'
            elif any(x in filename_lower for x in ['_bass', '_bassline']):
                return 'bass'
            else:
                return 'unknown'
            
    def compute_rms(self, sample):
        """Compute RMS of a sample (works for both mono and multi-channel, but expects 1D for mono)"""
        if sample.ndim > 1:
            # If somehow we get multi-channel here, convert to mono
            sample = np.mean(sample, axis=0) if sample.shape[0] < sample.shape[1] else np.mean(sample, axis=1)
        return np.sqrt(np.mean(sample**2))
    
    def extract_samples_from_stem(self, stem_audio, stem_type, sample_rate, metadata, file_hash):
        """
        Extract samples by scanning the entire stem for segments that meet the RMS requirement,
        then selecting sample regions and aligning to zero crossings. All processing in mono.
        """
        # Convert to mono immediately and ensure consistent format
        if stem_audio.ndim > 1:
            # Convert to proper mono - handle both (samples, channels) and (channels, samples)
            if stem_audio.shape[0] < stem_audio.shape[1]:  # (channels, samples)
                mono_audio = np.mean(stem_audio, axis=0)
            else:  # (samples, channels)
                mono_audio = np.mean(stem_audio, axis=1)
        else:
            mono_audio = stem_audio.copy()

        total_samples = len(mono_audio)
        sample_length = int(self.sample_duration * sample_rate)

        if total_samples < sample_length:
            print(f"Stem too short to extract samples: {total_samples/sample_rate:.2f}s")
            return []

        # --- SCAN THE ENTIRE AUDIO FOR LOUD ENOUGH CANDIDATES ---
        rms_window = sample_length
        hop = int(sample_length / 3)

        candidates = []
        idx = 0

        while idx + sample_length < total_samples:
            segment = mono_audio[idx:idx + sample_length]
            rms = self.compute_rms(segment)

            if rms >= self.min_rms:
                candidates.append(idx)
            idx += hop

        if not candidates:
            print(f"No loud-enough segments in {stem_type} stem")
            return []

        # Limit to required count
        np.random.shuffle(candidates)
        selected_starts = candidates[:self.samples_per_stem]
        samples = []

        for seq, start in enumerate(selected_starts):
            # Zero-crossing alignment on mono audio
            aligned_start = self.find_best_zero_crossing_around(mono_audio, start, sample_rate, window_duration=0.05)

            # Ensure bounds
            aligned_start = max(0, min(aligned_start, total_samples - sample_length))

            # Extract from mono audio for consistent processing
            sample_data = mono_audio[aligned_start:aligned_start + sample_length]

            actual_position = aligned_start / sample_rate

            samples.append({
                'stem_type': stem_type,
                'position': actual_position,
                'duration': self.sample_duration,
                'data': sample_data,  # This is now guaranteed mono
                'metadata': metadata.copy(),
                'file_hash': file_hash,
                'sequence': seq,
                'sample_rate': sample_rate,
                'rms': float(self.compute_rms(mono_audio[aligned_start:aligned_start+sample_length])),
            })

        print(f"Extracted {len(samples)} loud-enough samples from {stem_type} stem")
        return samples

    def find_zero_crossings(self, audio_mono, threshold=0.01):
        """
        Find meaningful zero-crossings in mono audio signal.
        
        Args:
            audio_mono: 1D numpy array of mono audio
            threshold: minimum amplitude to consider (to avoid noise)
        
        Returns:
            Array of indices where zero-crossings occur
        """
        # Remove DC offset
        audio_mono = audio_mono - np.mean(audio_mono)
        
        # Detect sign changes
        sign = np.sign(audio_mono)
        zero_crossings = np.where(np.diff(sign))[0]
        
        if len(zero_crossings) == 0:
            return zero_crossings
        
        # Filter crossings where amplitude is too low (avoid noise/null-zone)
        # Use the magnitude at the crossing point
        magnitudes = np.abs(audio_mono[zero_crossings])
        keep = magnitudes > threshold
        
        return zero_crossings[keep]

    def find_best_zero_crossing_around(self, audio_mono, target_sample, sample_rate, window_duration=0.05):
        """
        Find a zero-crossing close to target_sample in mono audio.
        
        Args:
            audio_mono: 1D numpy array of mono audio
            target_sample: the target sample index to align
            sample_rate: sample rate of the audio
            window_duration: search window in seconds
        
        Returns:
            Sample index aligned to the best zero-crossing
        """
        window = int(window_duration * sample_rate)
        
        start = max(0, target_sample - window)
        end = min(len(audio_mono), target_sample + window)
        
        segment = audio_mono[start:end]
        
        zero_crossings = self.find_zero_crossings(segment)
        
        if len(zero_crossings) == 0:
            return target_sample  # fallback
        
        # Find zero-crossing nearest to the original target position (relative to segment start)
        local_target = target_sample - start
        idx = np.argmin(np.abs(zero_crossings - local_target))
        return start + zero_crossings[idx]

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)


    def is_file_processed(self, file_hash):
        """Check if file has already been processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM samples WHERE file_hash = ?', (file_hash,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0

def main():
    parser = argparse.ArgumentParser(description='Build microsample database from music collection')
    parser.add_argument('input_dir', help='Root directory of music collection')
    parser.add_argument('--db-path', default='microsamples.db', help='SQLite database path')
    parser.add_argument('--samples-dir', default='samples', help='Output directory for audio samples')
    parser.add_argument('--temp-dir', default='temp_stems', help='Temporary directory for stem separation')
    parser.add_argument('--sample-duration', type=float, default=0.5, help='Sample duration in seconds')
    parser.add_argument('--samples-per-stem', type=int, default=10, help='Samples to extract per stem')
    parser.add_argument('--min-rms-db', type=int, default=-34, help='Minium RMS for extracted microsamples')
    
    args = parser.parse_args()
    
    config = {
        'db_path': args.db_path,
        'samples_output_dir': args.samples_dir,
        'temp_dir': args.temp_dir,
        'sample_duration': args.sample_duration,
        'samples_per_stem': args.samples_per_stem,
        'min_rms_db': args.min_rms_db
    }
    
    builder = MicrosampleDatabaseBuilder(config)

    try:
        builder.process_collection(args.input_dir)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        builder.cleanup()

if __name__ == "__main__":
    main()
import os
import sqlite3
import argparse
import numpy as np
from pathlib import Path
import tempfile
import shutil
from audio_separator.separator import Separator
import librosa
import soundfile as sf
from scipy import signal
import hashlib
import json
from datetime import datetime

class MicrosampleDatabaseBuilder:
    def __init__(self, config):
        self.config = config
        self.db_path = config['db_path']
        self.samples_output_dir = Path(config['samples_output_dir'])
        self.temp_dir = Path(config.get('temp_dir', 'temp_stems'))
        self.sample_duration = config.get('sample_duration', 0.5)  # seconds
        self.samples_per_stem = config.get('samples_per_stem', 10)
        
        # Create directories
        self.samples_output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
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

    def find_audio_files(self, root_dir):
        """Recursively find all MP3 and FLAC files in directory structure"""
        root_path = Path(root_dir)
        audio_files = []
        
        for ext in ['*.mp3', '*.flac', '*.wav', '*.m4a']:
            audio_files.extend(root_path.rglob(ext))
            
        print(f"Found {len(audio_files)} audio files in {root_dir}")
        return audio_files

    def extract_metadata_from_path(self, file_path):
        """Extract basic metadata from file path structure"""
        path_parts = file_path.parts
        # Simple heuristic: assume structure /artist/album/track.file
        if len(path_parts) >= 3:
            artist = path_parts[-3]
            album = path_parts[-2]
            song_title = Path(file_path).stem
        elif len(path_parts) == 2:
            artist = path_parts[-2]
            album = "Unknown Album"
            song_title = Path(file_path).stem
        else:
            artist = "Unknown Artist"
            album = "Unknown Album"
            song_title = Path(file_path).stem
            
        return {
            'artist': artist,
            'album': album,
            'song_title': song_title,
            'original_path': str(file_path)
        }

    def calculate_file_hash(self, file_path):
        """Calculate MD5 hash of file for unique identification"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def find_zero_crossings(self, audio_data, threshold=0.01):
        """Find zero-crossing points in audio data"""
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Remove DC offset
        audio_data = audio_data - np.mean(audio_data)
        
        # Find zero crossings where the signal changes sign
        zero_crossings = np.where(np.diff(np.sign(audio_data)))[0]
        
        # Filter by amplitude to avoid noise around zero
        if len(zero_crossings) > 0:
            magnitudes = np.abs(audio_data[zero_crossings])
            zero_crossings = zero_crossings[magnitudes > threshold]
        
        return zero_crossings

    def find_best_zero_crossing_around(self, audio_data, target_sample, sample_rate, window_duration=0.1):
        """Find the best zero-crossing point around target position"""
        window_samples = int(sample_rate * window_duration)
        start_search = max(0, target_sample - window_samples // 2)
        end_search = min(len(audio_data), target_sample + window_samples // 2)
        
        search_segment = audio_data[start_search:end_search]
        
        if len(search_segment) == 0:
            return target_sample
            
        zero_crossings = self.find_zero_crossings(search_segment)
        
        if len(zero_crossings) == 0:
            return target_sample
            
        # Find closest zero crossing to target
        local_target = target_sample - start_search
        closest_idx = np.argmin(np.abs(zero_crossings - local_target))
        
        return start_search + zero_crossings[closest_idx]

    def extract_samples_from_stem(self, stem_audio, stem_type, sample_rate, metadata, file_hash):
        """Extract multiple samples from a stem with zero-crossing alignment"""
        # Ensure audio is in the right shape for processing
        if len(stem_audio.shape) == 1:
            # Mono audio
            audio_data = stem_audio
        else:
            # Multi-channel, convert to mono for processing
            audio_data = np.mean(stem_audio, axis=0) if stem_audio.shape[0] < stem_audio.shape[1] else np.mean(stem_audio, axis=1)
        
        total_duration = len(audio_data) / sample_rate
        
        if total_duration <= self.sample_duration:
            print(f"Stem too short for sampling: {total_duration:.2f}s")
            return []
        
        # Calculate possible sample positions (avoid very beginning and end)
        margin = self.sample_duration * 2
        available_duration = total_duration - margin * 2
        
        if available_duration <= 0:
            print(f"Not enough duration for sampling after margins: {total_duration:.2f}s")
            return []
        
        sample_positions = np.linspace(margin, total_duration - margin, self.samples_per_stem)
        
        samples = []
        sample_length = int(self.sample_duration * sample_rate)
        
        for i, pos_seconds in enumerate(sample_positions):
            target_sample = int(pos_seconds * sample_rate)
            
            # Find optimal zero-crossing point
            optimal_start = self.find_best_zero_crossing_around(audio_data, target_sample, sample_rate)
            
            # Ensure we don't go out of bounds
            if optimal_start + sample_length > len(audio_data):
                optimal_start = len(audio_data) - sample_length
            if optimal_start < 0:
                continue
            
            # Extract sample from original stem audio (preserving channels)
            if len(stem_audio.shape) == 1:
                sample_data = stem_audio[optimal_start:optimal_start + sample_length]
            else:
                if stem_audio.shape[0] < stem_audio.shape[1]:  # (channels, samples)
                    sample_data = stem_audio[:, optimal_start:optimal_start + sample_length]
                else:  # (samples, channels)
                    sample_data = stem_audio[optimal_start:optimal_start + sample_length, :]
            
            actual_duration = len(sample_data) / sample_rate if len(sample_data.shape) == 1 else sample_data.shape[0] / sample_rate
            actual_position = optimal_start / sample_rate
            
            sample_info = {
                'stem_type': stem_type,
                'position': actual_position,
                'duration': actual_duration,
                'data': sample_data,
                'metadata': metadata.copy(),
                'file_hash': file_hash,
                'sequence': i,
                'sample_rate': sample_rate
            }
            samples.append(sample_info)
            
        print(f"Extracted {len(samples)} samples from {stem_type} stem")
        return samples

    def separate_audio_stems(self, audio_file):
        """Separate audio file into stems using python-audio-separator"""
        try:
            print(f"Starting stem separation for: {audio_file}")
            
            # Initialize separator with explicit settings
            separator = Separator()
            
            # Configure separator with explicit paths
            separator.output_dir = str(self.temp_dir)
            separator.model_file_dir = str(self.temp_dir / "models")
            
            # Create directories if they don't exist
            os.makedirs(separator.output_dir, exist_ok=True)
            os.makedirs(separator.model_file_dir, exist_ok=True)
            
            print(f"Output directory: {separator.output_dir}")
            print(f"Model directory: {separator.model_file_dir}")
            
            # List available models in the directory
            model_path = Path(separator.model_file_dir)
            if model_path.exists():
                model_files = list(model_path.glob("*.onnx")) + list(model_path.glob("*.pth"))
                if model_files:
                    print("Found existing model files:")
                    for mf in model_files:
                        print(f"  - {mf.name}")
                else:
                    print("No model files found. Will download on first run.")
            
            # Try with a specific model known to work well
            # You can change this to other models like 'UVR_MDXNET_KARA_2.onnx'
            separator.model_filename = 'htdemucs_ft.yaml'  # Demucs model that separates 4 stems
            
            print(f"Using model: {separator.model_filename}")
            print("Starting audio separation (this may download models on first run)...")
            
            # Perform separation
            separator.load_model()
            output_files = separator.separate(str(audio_file))
            
            if not output_files:
                print("No output files generated from separation")
                return {}
            
            print(f"Separation completed. Generated {len(output_files)} stem files:")
            for output_file in output_files:
                print(f"  - {output_file}")
            
            stems = {}
            for output_file in output_files:
                stem_type = self.determine_stem_type(output_file)
                print(f"Loading {stem_type} stem from: {output_file}")
                
                try:
                    # Load the separated audio
                    audio_data, sample_rate = librosa.load(output_file, sr=None, mono=False)
                    stems[stem_type] = {
                        'audio': audio_data,
                        'sample_rate': sample_rate,
                        'file_path': output_file
                    }
                    print(f"Successfully loaded {stem_type} stem")
                except Exception as e:
                    print(f"Error loading stem {output_file}: {e}")
            
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

    def save_sample_to_disk(self, sample_data, sample_rate, sample_id):
        """Save sample audio to disk"""
        filename = f"{sample_id}.wav"
        filepath = self.samples_output_dir / filename
        
        try:
            # Ensure the data is in correct format for soundfile
            if len(sample_data.shape) == 1:
                # Mono audio
                sample_data_to_save = sample_data
            elif sample_data.shape[0] == 2 and sample_data.shape[1] > 2:
                # Shape: (channels, samples) -> convert to (samples, channels)
                sample_data_to_save = sample_data.T
            else:
                # Assume (samples, channels) or other format
                sample_data_to_save = sample_data
            
            sf.write(filepath, sample_data_to_save, sample_rate)
            return filename
        except Exception as e:
            print(f"Error saving sample {sample_id}: {e}")
            # Fallback: try with mono conversion
            try:
                if len(sample_data.shape) > 1:
                    sample_data = np.mean(sample_data, axis=0)
                sf.write(filepath, sample_data, sample_rate)
                return filename
            except Exception as e2:
                print(f"Fallback save also failed: {e2}")
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
            return True
            
        except Exception as e:
            print(f"✗ Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def is_file_processed(self, file_hash):
        """Check if file has already been processed"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM samples WHERE file_hash = ?', (file_hash,))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count > 0

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

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Build microsample database from music collection')
    parser.add_argument('input_dir', help='Root directory of music collection')
    parser.add_argument('--db-path', default='microsamples.db', help='SQLite database path')
    parser.add_argument('--samples-dir', default='samples', help='Output directory for audio samples')
    parser.add_argument('--temp-dir', default='temp_stems', help='Temporary directory for stem separation')
    parser.add_argument('--sample-duration', type=float, default=0.5, help='Sample duration in seconds')
    parser.add_argument('--samples-per-stem', type=int, default=10, help='Samples to extract per stem')
    
    args = parser.parse_args()
    
    config = {
        'db_path': args.db_path,
        'samples_output_dir': args.samples_dir,
        'temp_dir': args.temp_dir,
        'sample_duration': args.sample_duration,
        'samples_per_stem': args.samples_per_stem
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
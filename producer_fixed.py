import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """Handles audio I/O and processing operations"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            raise ValueError(f"Could not load audio file {file_path}: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str, sr: int = None):
        """Save audio data to file"""
        if sr is None:
            sr = self.sample_rate
        sf.write(file_path, audio, sr)
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Extract comprehensive audio features for analysis"""
        features = {}
        
        # Timbral features
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # Harmonic features
        features['chroma'] = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Rhythmic features
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        features['beat_frames'] = beat_frames
        features['onset_strength'] = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Spectral features
        features['spectral_contrast'] = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['mel_spectrogram'] = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        return features
    
    def detect_beats(self, audio: np.ndarray, sr: int) -> Dict:
        """Detect beats and tempo in audio"""
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        return {
            'tempo': tempo,
            'beat_times': beat_times,
            'beat_frames': beat_frames,
            'onset_times': onset_times,
            'onset_frames': onset_frames
        }

class Sampler:
    """Advanced audio sampling capabilities"""
    
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
    
    def extract_sample(self, audio: np.ndarray, sr: int, start_time: float, duration: float) -> np.ndarray:
        """Extract a sample from audio at specified time and duration"""
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)
        return audio[start_sample:end_sample]
    
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Time-stretch audio without changing pitch"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
        """Pitch-shift audio without changing tempo"""
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def create_loop(self, audio: np.ndarray, sr: int, loop_duration: float) -> np.ndarray:
        """Create a seamless loop from audio"""
        beat_info = self.audio_processor.detect_beats(audio, sr)
        beat_times = beat_info['beat_times']
        
        # Find beats that fit within loop duration
        valid_beats = beat_times[beat_times <= loop_duration]
        if len(valid_beats) < 2:
            # Fallback to simple duration-based loop
            loop_samples = int(loop_duration * sr)
            return audio[:loop_samples]
        
        # Use beat-aligned loop
        loop_end_time = valid_beats[-1]
        loop_samples = int(loop_end_time * sr)
        return audio[:loop_samples]

class StyleRecognizer:
    """Recognizes and classifies artist styles"""
    
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        
        # Artist style characteristics
        self.artist_profiles = {
            'travis_scott': {
                'tempo_range': (130, 160),
                'vocal_characteristics': ['autotune_heavy', 'adlibs_frequent'],
                'production_style': ['atmospheric', 'heavy_bass', '808_drums']
            },
            'kanye_west': {
                'tempo_range': (80, 140),
                'vocal_characteristics': ['soul_samples', 'vocal_chops'],
                'production_style': ['orchestral', 'gospel_influenced', 'innovative']
            },
            'metro_boomin': {
                'tempo_range': (130, 150),
                'vocal_characteristics': ['trap_influenced'],
                'production_style': ['dark_atmosphere', 'heavy_808s', 'minimalist']
            }
        }
    
    def flatten_features(self, features: Dict) -> np.ndarray:
        """Flatten feature dictionary into a single vector"""
        feature_vector = []
        
        # Add statistical measures of each feature
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                if value.ndim > 1:
                    # For 2D features, take mean across time axis
                    feature_vector.extend([
                        np.mean(value),
                        np.std(value),
                        np.max(value),
                        np.min(value)
                    ])
                else:
                    # For 1D features
                    feature_vector.extend([
                        np.mean(value),
                        np.std(value)
                    ])
            else:
                # For scalar features
                feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def analyze_style(self, audio: np.ndarray, sr: int) -> Dict:
        """Analyze audio and return style characteristics"""
        features = self.audio_processor.extract_features(audio, sr)
        beat_info = self.audio_processor.detect_beats(audio, sr)
        
        style_analysis = {
            'tempo': beat_info['tempo'],
            'energy': np.mean(features['spectral_centroid']),
            'brightness': np.mean(features['spectral_rolloff']),
            'rhythmic_complexity': len(beat_info['onset_times']) / (len(audio) / sr),
            'harmonic_richness': np.mean(features['chroma']),
            'timbral_characteristics': {
                'mfcc_mean': np.mean(features['mfcc'], axis=1),
                'spectral_contrast_mean': np.mean(features['spectral_contrast'], axis=1)
            }
        }
        
        return style_analysis
    
    def predict_artist_style(self, audio: np.ndarray, sr: int) -> str:
        """Predict which artist style the audio most resembles"""
        if not self.is_trained:
            return self._rule_based_classification(audio, sr)
        
        features = self.audio_processor.extract_features(audio, sr)
        feature_vector = self.flatten_features(features).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        prediction = self.classifier.predict(feature_vector_scaled)[0]
        return prediction
    
    def _rule_based_classification(self, audio: np.ndarray, sr: int) -> str:
        """Rule-based classification when no trained model is available"""
        style_analysis = self.analyze_style(audio, sr)
        tempo = style_analysis['tempo']
        
        # Simple rule-based classification
        if 130 <= tempo <= 160 and style_analysis['energy'] > 1000:
            return 'travis_scott'
        elif 80 <= tempo <= 140 and style_analysis['harmonic_richness'] > 0.5:
            return 'kanye_west'
        elif 130 <= tempo <= 150 and style_analysis['brightness'] > 2000:
            return 'metro_boomin'
        else:
            return 'unknown'

class BeatGenerator:
    """Generates beats and drum patterns"""
    
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
        
        # Basic drum samples (in a real implementation, these would be actual audio files)
        self.drum_patterns = {
            'travis_scott': {
                'kick_pattern': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                'snare_pattern': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_pattern': [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
                'tempo': 145
            },
            'kanye_west': {
                'kick_pattern': [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                'snare_pattern': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_pattern': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                'tempo': 120
            },
            'metro_boomin': {
                'kick_pattern': [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                'snare_pattern': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                'hihat_pattern': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                'tempo': 140
            }
        }
    
    def generate_beat_pattern(self, style: str, bars: int = 4) -> Dict:
        """Generate a beat pattern in the specified style"""
        if style not in self.drum_patterns:
            style = 'travis_scott'  # Default fallback
        
        pattern = self.drum_patterns[style]
        
        # Extend pattern for multiple bars
        beat_pattern = {
            'kick': pattern['kick_pattern'] * bars,
            'snare': pattern['snare_pattern'] * bars,
            'hihat': pattern['hihat_pattern'] * bars,
            'tempo': pattern['tempo'],
            'bars': bars
        }
        
        return beat_pattern
    
    def create_beat_audio(self, pattern: Dict, duration: float = 8.0) -> np.ndarray:
        """Create audio from beat pattern (simplified version)"""
        # This is a simplified implementation
        # In a real version, you'd use actual drum samples
        
        sr = 22050
        samples = int(duration * sr)
        audio = np.zeros(samples)
        
        tempo = pattern['tempo']
        beat_duration = 60.0 / tempo / 4  # Duration of each 16th note
        
        # Add basic synthesized drum sounds
        for i, kick in enumerate(pattern['kick']):
            if kick and i * beat_duration < duration:
                # Simple kick drum synthesis (low frequency sine wave)
                t = np.linspace(0, 0.1, int(0.1 * sr))
                kick_sound = 0.5 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 20)
                start_idx = int(i * beat_duration * sr)
                end_idx = min(start_idx + len(kick_sound), len(audio))
                audio[start_idx:end_idx] += kick_sound[:end_idx-start_idx]
        
        return audio

class AIProducer:
    """Main AI Producer orchestrating all components"""
    
    def __init__(self, sample_rate: int = 22050):
        self.audio_processor = AudioProcessor(sample_rate)
        self.sampler = Sampler(self.audio_processor)
        self.style_recognizer = StyleRecognizer(self.audio_processor)
        self.beat_generator = BeatGenerator(self.audio_processor)
    
    def analyze_sample(self, file_path: str) -> Dict:
        """Analyze an audio sample and return comprehensive information"""
        audio, sr = self.audio_processor.load_audio(file_path)
        
        features = self.audio_processor.extract_features(audio, sr)
        beat_info = self.audio_processor.detect_beats(audio, sr)
        style_analysis = self.style_recognizer.analyze_style(audio, sr)
        predicted_style = self.style_recognizer.predict_artist_style(audio, sr)
        
        return {
            'file_path': file_path,
            'duration': len(audio) / sr,
            'sample_rate': sr,
            'tempo': beat_info['tempo'],
            'predicted_style': predicted_style,
            'style_characteristics': style_analysis,
            'beat_count': len(beat_info['beat_times']),
            'features_summary': {
                'avg_spectral_centroid': np.mean(features['spectral_centroid']),
                'avg_mfcc': np.mean(features['mfcc']),
                'harmonic_content': np.mean(features['chroma'])
            }
        }
    
    def create_beat(self, style: str, duration: float = 8.0, bars: int = 4) -> Tuple[np.ndarray, Dict]:
        """Create a beat in the specified artist style"""
        pattern = self.beat_generator.generate_beat_pattern(style, bars)
        audio = self.beat_generator.create_beat_audio(pattern, duration)
        
        return audio, pattern
    
    def sample_and_remix(self, source_file: str, target_style: str, output_file: str):
        """Sample audio and remix it in target artist style"""
        # Load and analyze source
        source_audio, sr = self.audio_processor.load_audio(source_file)
        source_analysis = self.analyze_sample(source_file)
        
        # Extract interesting samples
        beat_info = self.audio_processor.detect_beats(source_audio, sr)
        beat_times = beat_info['beat_times']
        
        # Create samples between beats
        samples = []
        for i in range(min(4, len(beat_times)-1)):
            start_time = beat_times[i]
            end_time = beat_times[i+1]
            sample = self.sampler.extract_sample(source_audio, sr, start_time, end_time - start_time)
            samples.append(sample)
        
        # Generate beat in target style
        beat_audio, pattern = self.create_beat(target_style)
        
        # Simple remix: layer samples over beat
        remix = beat_audio.copy()
        for i, sample in enumerate(samples):
            if len(sample) > 0:
                # Place sample at different points in the beat
                placement_time = i * 2.0  # Every 2 seconds
                placement_idx = int(placement_time * sr)
                
                if placement_idx + len(sample) < len(remix):
                    # Adjust sample volume and add to remix
                    sample_normalized = sample * 0.3  # Reduce volume
                    remix[placement_idx:placement_idx + len(sample)] += sample_normalized
        
        # Save result
        self.audio_processor.save_audio(remix, output_file, sr)
        
        return {
            'source_analysis': source_analysis,
            'target_style': target_style,
            'output_file': output_file,
            'beat_pattern': pattern
        }

def main():
    """Example usage of the AI Producer"""
    producer = AIProducer()
    
    print("AI Producer initialized!")
    print("Available artist styles: travis_scott, kanye_west, metro_boomin")
    
    # Example: Create a beat
    print("\nCreating a Travis Scott type beat...")
    beat_audio, pattern = producer.create_beat('travis_scott', duration=8.0)
    producer.audio_processor.save_audio(beat_audio, 'travis_scott_beat.wav')
    print(f"Beat created with tempo: {pattern['tempo']} BPM")
    
    print("\nAI Producer ready for sampling and beat creation!")

if __name__ == "__main__":
    main()
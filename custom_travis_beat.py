#!/usr/bin/env python3
"""
Custom Travis Scott Style Beat Generator
Creates moody, atmospheric trap beats with psychedelic elements
"""

from producer_fixed import AIProducer
import numpy as np

def create_atmospheric_travis_beat():
    """Create a custom atmospheric Travis Scott style beat"""
    
    producer = AIProducer()
    
    print("Creating moody, atmospheric Travis Scott beat...")
    print("Features: Psychedelic textures, ambient synths, hard-hitting drums")
    
    # Generate base Travis Scott beat
    beat_audio, pattern = producer.create_beat('travis_scott', duration=16.0, bars=8)
    
    # Enhanced parameters for atmospheric feel
    sr = 22050
    duration = 16.0
    samples = int(duration * sr)
    
    # Create enhanced atmospheric version
    atmospheric_beat = np.zeros(samples)
    
    # Base drum pattern (Travis Scott style)
    tempo = 145
    beat_duration = 60.0 / tempo / 4  # Duration of each 16th note
    
    # Enhanced kick pattern with variations
    kick_pattern = [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0] * 8
    snare_pattern = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * 8
    hihat_pattern = [1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1] * 8
    
    # Add atmospheric elements
    for i in range(len(kick_pattern)):
        time_offset = i * beat_duration
        
        if time_offset >= duration:
            break
            
        start_idx = int(time_offset * sr)
        
        # Enhanced kick drum with sub-bass
        if kick_pattern[i]:
            # Main kick (60Hz fundamental)
            t = np.linspace(0, 0.15, int(0.15 * sr))
            kick_sound = 0.7 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 15)
            
            # Add sub-bass component (30Hz)
            sub_bass = 0.4 * np.sin(2 * np.pi * 30 * t) * np.exp(-t * 10)
            kick_sound += sub_bass
            
            end_idx = min(start_idx + len(kick_sound), len(atmospheric_beat))
            atmospheric_beat[start_idx:end_idx] += kick_sound[:end_idx-start_idx]
        
        # Enhanced snare with ambient reverb tail
        if snare_pattern[i]:
            # Main snare hit
            t = np.linspace(0, 0.1, int(0.1 * sr))
            noise = np.random.normal(0, 1, len(t))
            snare_sound = 0.5 * noise * np.exp(-t * 20)
            
            # Add reverb tail for atmosphere
            reverb_t = np.linspace(0, 0.5, int(0.5 * sr))
            reverb_tail = 0.2 * np.random.normal(0, 1, len(reverb_t)) * np.exp(-reverb_t * 5)
            
            # Combine snare and reverb
            full_snare = np.concatenate([snare_sound, reverb_tail])
            
            end_idx = min(start_idx + len(full_snare), len(atmospheric_beat))
            atmospheric_beat[start_idx:end_idx] += full_snare[:end_idx-start_idx]
        
        # Hi-hats with stereo width simulation
        if hihat_pattern[i]:
            t = np.linspace(0, 0.05, int(0.05 * sr))
            hihat_sound = 0.3 * np.random.normal(0, 1, len(t)) * np.exp(-t * 40)
            
            end_idx = min(start_idx + len(hihat_sound), len(atmospheric_beat))
            atmospheric_beat[start_idx:end_idx] += hihat_sound[:end_idx-start_idx]
    
    # Add ambient synth layers
    print("Adding ambient synth layers...")
    
    # Low-frequency ambient pad
    t_full = np.linspace(0, duration, samples)
    
    # Pad progression (atmospheric chords)
    pad_freqs = [110, 146.83, 174.61, 220]  # A, D, F, A progression
    
    for i, freq in enumerate(pad_freqs):
        # Each chord lasts 4 seconds
        start_time = i * 4
        end_time = min((i + 1) * 4, duration)
        
        if start_time >= duration:
            break
            
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        chord_length = end_sample - start_sample
        
        t_chord = np.linspace(0, end_time - start_time, chord_length)
        
        # Create atmospheric pad sound
        pad_sound = (0.15 * np.sin(2 * np.pi * freq * t_chord) * 
                    np.exp(-t_chord * 0.5) +
                    0.1 * np.sin(2 * np.pi * freq * 1.5 * t_chord) * 
                    np.exp(-t_chord * 0.3))
        
        # Add slow attack for atmosphere
        attack_samples = int(0.5 * sr)  # 0.5 second attack
        if len(pad_sound) > attack_samples:
            pad_sound[:attack_samples] *= np.linspace(0, 1, attack_samples)
        
        atmospheric_beat[start_sample:end_sample] += pad_sound
    
    # Add psychedelic texture layers
    print("Adding psychedelic textures...")
    
    # Modulated sine wave for psychedelic effect
    modulation_freq = 0.3  # Slow modulation
    carrier_freq = 440  # A note
    
    psychedelic_texture = (0.1 * np.sin(2 * np.pi * carrier_freq * t_full) * 
                          (0.5 + 0.5 * np.sin(2 * np.pi * modulation_freq * t_full)))
    
    # Apply filtering effect (simple lowpass simulation)
    psychedelic_texture = np.convolve(psychedelic_texture, 
                                    np.ones(50)/50, mode='same')
    
    atmospheric_beat += psychedelic_texture
    
    # Add spacious reverb effect (simple delay)
    print("Adding spacious reverb...")
    
    delay_samples = int(0.25 * sr)  # 250ms delay
    reverb_level = 0.3
    
    if len(atmospheric_beat) > delay_samples:
        delayed_signal = np.zeros_like(atmospheric_beat)
        delayed_signal[delay_samples:] = atmospheric_beat[:-delay_samples] * reverb_level
        atmospheric_beat += delayed_signal
    
    # Normalize to prevent clipping
    max_amplitude = np.max(np.abs(atmospheric_beat))
    if max_amplitude > 0:
        atmospheric_beat = atmospheric_beat / max_amplitude * 0.8
    
    # Save the atmospheric beat
    output_filename = "atmospheric_travis_scott_beat.wav"
    producer.audio_processor.save_audio(atmospheric_beat, output_filename)
    
    print(f"Created: {output_filename}")
    print("Characteristics:")
    print("- Tempo: 145 BPM (Travis Scott style)")
    print("- Mood: Moody and atmospheric")
    print("- Features: Psychedelic textures, ambient synth pads")
    print("- Drums: Hard-hitting with sub-bass and reverb tails")
    print("- Style: Cinematic, spacious, hypnotic")
    print("- Perfect for: Auto-tuned vocal flows")
    
    return atmospheric_beat, {
        'tempo': 145,
        'style': 'travis_scott_atmospheric',
        'duration': duration,
        'features': ['psychedelic_textures', 'ambient_synths', 'hard_drums', 'spacious_reverb']
    }

def main():
    print("Custom Atmospheric Travis Scott Beat Generator")
    print("=" * 60)
    
    # Create the atmospheric beat
    beat, info = create_atmospheric_travis_beat()
    
    print(f"\nBeat generation complete!")
    print(f"Duration: {info['duration']} seconds")
    print(f"Style: {info['style']}")
    print(f"Features: {', '.join(info['features'])}")
    
    print(f"\nThis beat incorporates:")
    print("✓ Moody atmospheric elements")
    print("✓ Psychedelic textures with modulation")
    print("✓ Ambient synth layers and pads") 
    print("✓ Hard-hitting drums with sub-bass")
    print("✓ Spacious reverb for cinematic feel")
    print("✓ Hypnotic, repetitive patterns")
    print("✓ Perfect spacing for auto-tuned vocals")

if __name__ == "__main__":
    main()
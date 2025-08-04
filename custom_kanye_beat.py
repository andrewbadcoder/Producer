#!/usr/bin/env python3
"""
Custom Kanye West Style Beat Generator
Creates bold, soulful, experimental hip-hop beats with chopped samples and creative arrangements
"""

from producer_fixed import AIProducer
import numpy as np

def create_experimental_kanye_beat():
    """Create a custom experimental Kanye West style beat"""
    
    producer = AIProducer()
    
    print("Creating bold, soulful Kanye West beat...")
    print("Features: Chopped soul samples, creative drums, unconventional textures")
    
    # Enhanced parameters for soulful feel
    sr = 22050
    duration = 20.0  # Extended for dynamic arrangements
    samples = int(duration * sr)
    
    # Create experimental beat
    experimental_beat = np.zeros(samples)
    
    # Kanye-style drum pattern (more complex and dynamic)
    tempo = 95  # Slower, more soulful tempo
    beat_duration = 60.0 / tempo / 4  # Duration of each 16th note
    
    # Dynamic drum patterns that change throughout
    # Section A (0-8s): Classic boom-bap
    kick_pattern_a = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0] * 2
    snare_pattern_a = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] * 2
    
    # Section B (8-16s): More complex, syncopated
    kick_pattern_b = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0] * 2
    snare_pattern_b = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1] * 2
    
    # Section C (16-20s): Breakdown with minimal drums
    kick_pattern_c = [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    snare_pattern_c = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    
    # Combine patterns for dynamic arrangement
    all_kick_patterns = kick_pattern_a + kick_pattern_b + kick_pattern_c
    all_snare_patterns = snare_pattern_a + snare_pattern_b + snare_pattern_c
    
    print("Programming dynamic drum arrangements...")
    
    # Add drums with dynamic processing
    for i in range(len(all_kick_patterns)):
        time_offset = i * beat_duration
        
        if time_offset >= duration:
            break
            
        start_idx = int(time_offset * sr)
        
        # Soulful kick with warmth
        if all_kick_patterns[i]:
            # Warm, punchy kick (80Hz fundamental)
            t = np.linspace(0, 0.12, int(0.12 * sr))
            kick_sound = 0.8 * np.sin(2 * np.pi * 80 * t) * np.exp(-t * 12)
            
            # Add subtle harmonics for character
            kick_sound += 0.3 * np.sin(2 * np.pi * 160 * t) * np.exp(-t * 18)
            
            # Vary intensity based on section
            section = int(time_offset / 8)  # 0, 1, or 2
            if section == 2:  # Breakdown section
                kick_sound *= 0.6  # Quieter
            
            end_idx = min(start_idx + len(kick_sound), len(experimental_beat))
            experimental_beat[start_idx:end_idx] += kick_sound[:end_idx-start_idx]
        
        # Creative snare with character
        if i < len(all_snare_patterns) and all_snare_patterns[i]:
            # Layered snare with crack and body
            t = np.linspace(0, 0.08, int(0.08 * sr))
            
            # Main snare body (200Hz)
            snare_body = 0.4 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 15)
            
            # High-frequency crack
            noise = np.random.normal(0, 1, len(t))
            snare_crack = 0.6 * noise * np.exp(-t * 25)
            
            snare_sound = snare_body + snare_crack
            
            # Vary snare character by section
            section = int(time_offset / 8)
            if section == 1:  # Complex section - add reverb
                reverb_t = np.linspace(0, 0.3, int(0.3 * sr))
                reverb = 0.3 * np.random.normal(0, 1, len(reverb_t)) * np.exp(-reverb_t * 8)
                snare_sound = np.concatenate([snare_sound, reverb])
            
            end_idx = min(start_idx + len(snare_sound), len(experimental_beat))
            experimental_beat[start_idx:end_idx] += snare_sound[:end_idx-start_idx]
    
    # Add chopped soul sample simulation
    print("Adding chopped soul sample elements...")
    
    # Create soul chord progression (simulating chopped samples)
    soul_chords = [
        [220, 277.18, 329.63],    # A minor
        [196, 246.94, 293.66],    # G major  
        [174.61, 220, 261.63],    # F major
        [146.83, 185, 220]        # D minor
    ]
    
    t_full = np.linspace(0, duration, samples)
    
    # Create chopped sample effect with different sections
    for section in range(3):  # 3 sections of the beat
        section_start = section * (duration / 3)
        section_end = (section + 1) * (duration / 3)
        
        # Different chopping pattern per section
        if section == 0:  # Smooth, sustained chords
            chop_duration = 2.0
        elif section == 1:  # More chopped, rhythmic
            chop_duration = 0.5
        else:  # Minimal, sparse
            chop_duration = 4.0
        
        current_time = section_start
        chord_idx = 0
        
        while current_time < section_end:
            if chord_idx >= len(soul_chords):
                chord_idx = 0
            
            chord = soul_chords[chord_idx]
            
            # Calculate sample indices
            start_sample = int(current_time * sr)
            end_time = min(current_time + chop_duration, section_end)
            end_sample = int(end_time * sr)
            
            if start_sample >= len(experimental_beat):
                break
                
            chord_length = end_sample - start_sample
            t_chord = np.linspace(0, end_time - current_time, chord_length)
            
            # Create rich, warm chord sound
            chord_sound = np.zeros(chord_length)
            for freq in chord:
                # Multiple harmonics for richness
                chord_sound += (0.2 * np.sin(2 * np.pi * freq * t_chord) +
                              0.15 * np.sin(2 * np.pi * freq * 2 * t_chord) +
                              0.1 * np.sin(2 * np.pi * freq * 3 * t_chord))
            
            # Add subtle vibrato for soul character
            vibrato = 1 + 0.02 * np.sin(2 * np.pi * 4 * t_chord)
            chord_sound *= vibrato
            
            # Apply envelope based on section
            if section == 1:  # Chopped section - sharp attack/decay
                attack_samples = int(0.01 * sr)  # Quick attack
                decay_samples = int(0.1 * sr)   # Quick decay
                if len(chord_sound) > attack_samples:
                    chord_sound[:attack_samples] *= np.linspace(0, 1, attack_samples)
                if len(chord_sound) > decay_samples:
                    chord_sound[-decay_samples:] *= np.linspace(1, 0, decay_samples)
            else:  # Smooth sections
                attack_samples = int(0.1 * sr)
                if len(chord_sound) > attack_samples:
                    chord_sound[:attack_samples] *= np.linspace(0, 1, attack_samples)
            
            # Add to main beat
            end_idx = min(start_sample + len(chord_sound), len(experimental_beat))
            experimental_beat[start_sample:end_idx] += chord_sound[:end_idx-start_sample]
            
            current_time += chop_duration
            chord_idx += 1
    
    # Add unconventional textures
    print("Adding unconventional textures and creative elements...")
    
    # Texture 1: Filtered noise sweeps
    noise_sweep = np.random.normal(0, 1, samples) * 0.1
    
    # Create filter sweep effect
    for i in range(samples):
        t_norm = i / samples
        # Varying filter effect (simple amplitude modulation)
        filter_amount = 0.5 + 0.5 * np.sin(2 * np.pi * 0.1 * t_norm)
        noise_sweep[i] *= filter_amount
    
    experimental_beat += noise_sweep
    
    # Texture 2: Pitched vocal-like sounds (abstract, not actual vocals)
    vocal_texture = np.zeros(samples)
    for i in range(0, samples, int(2 * sr)):  # Every 2 seconds
        if i + int(0.5 * sr) < samples:
            t_vocal = np.linspace(0, 0.5, int(0.5 * sr))
            # Create abstract vocal-like formant
            formant = (0.15 * np.sin(2 * np.pi * 440 * t_vocal) * 
                      np.sin(2 * np.pi * 8 * t_vocal) * 
                      np.exp(-t_vocal * 3))
            vocal_texture[i:i+len(formant)] += formant
    
    experimental_beat += vocal_texture
    
    # Add creative panning simulation (subtle left-right movement)
    print("Adding spatial movement and final processing...")
    
    # Simple stereo width simulation through phase modulation
    phase_mod = 0.05 * np.sin(2 * np.pi * 0.2 * t_full)
    experimental_beat += phase_mod
    
    # Dynamic range compression simulation
    # Soft limiting to add punch
    threshold = 0.7
    ratio = 3.0
    
    for i in range(len(experimental_beat)):
        if abs(experimental_beat[i]) > threshold:
            excess = abs(experimental_beat[i]) - threshold
            compressed_excess = excess / ratio
            experimental_beat[i] = np.sign(experimental_beat[i]) * (threshold + compressed_excess)
    
    # Final normalization
    max_amplitude = np.max(np.abs(experimental_beat))
    if max_amplitude > 0:
        experimental_beat = experimental_beat / max_amplitude * 0.85
    
    # Save the experimental beat
    output_filename = "experimental_kanye_west_beat.wav"
    producer.audio_processor.save_audio(experimental_beat, output_filename)
    
    print(f"Created: {output_filename}")
    print("Characteristics:")
    print("- Tempo: 95 BPM (soulful, contemplative)")
    print("- Mood: Bold and experimental with emotional depth")
    print("- Arrangement: Dynamic with 3 distinct sections")
    print("- Samples: Chopped soul chord progressions")
    print("- Drums: Creative programming with varying intensity")
    print("- Textures: Unconventional elements and spatial movement")
    print("- Style: Perfect for expressive, thought-provoking lyrics")
    
    return experimental_beat, {
        'tempo': 95,
        'style': 'kanye_west_experimental',
        'duration': duration,
        'sections': 3,
        'features': ['chopped_soul_samples', 'dynamic_arrangements', 'creative_drums', 'unconventional_textures']
    }

def main():
    print("Custom Experimental Kanye West Beat Generator")
    print("=" * 65)
    
    # Create the experimental beat
    beat, info = create_experimental_kanye_beat()
    
    print(f"\nBeat generation complete!")
    print(f"Duration: {info['duration']} seconds")
    print(f"Style: {info['style']}")
    print(f"Sections: {info['sections']} dynamic arrangements")
    print(f"Features: {', '.join(info['features'])}")
    
    print(f"\nThis beat incorporates:")
    print("- Bold, soulful foundation with emotional depth")
    print("- Chopped soul sample simulation with rich harmonics")
    print("- Creative drum programming across 3 sections")
    print("- Unconventional textures and spatial movement")
    print("- Dynamic arrangements perfect for storytelling")
    print("- Experimental elements true to Kanye's innovative style")

if __name__ == "__main__":
    main()
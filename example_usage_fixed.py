#!/usr/bin/env python3
"""
Example usage of the AI Producer
Run this script to see the AI Producer in action
"""

from producer_fixed import AIProducer
import os

def main():
    print("AI Producer - Example Usage")
    print("=" * 50)
    
    # Initialize the AI Producer
    producer = AIProducer()
    
    print("\n1. Creating beats in different artist styles...")
    
    # Create Travis Scott type beat
    print("Creating Travis Scott type beat...")
    travis_beat, travis_pattern = producer.create_beat('travis_scott', duration=8.0)
    producer.audio_processor.save_audio(travis_beat, 'travis_scott_beat.wav')
    print(f"   Travis Scott beat created (Tempo: {travis_pattern['tempo']} BPM)")
    
    # Create Kanye West type beat  
    print("Creating Kanye West type beat...")
    kanye_beat, kanye_pattern = producer.create_beat('kanye_west', duration=8.0)
    producer.audio_processor.save_audio(kanye_beat, 'kanye_west_beat.wav')
    print(f"   Kanye West beat created (Tempo: {kanye_pattern['tempo']} BPM)")
    
    # Create Metro Boomin type beat
    print("Creating Metro Boomin type beat...")
    metro_beat, metro_pattern = producer.create_beat('metro_boomin', duration=8.0)
    producer.audio_processor.save_audio(metro_beat, 'metro_boomin_beat.wav')
    print(f"   Metro Boomin beat created (Tempo: {metro_pattern['tempo']} BPM)")
    
    print("\n2. Beat pattern analysis...")
    print(f"Travis Scott pattern: {travis_pattern['kick'][:8]}... (showing first 8 steps)")
    print(f"Kanye West pattern:   {kanye_pattern['kick'][:8]}... (showing first 8 steps)")
    print(f"Metro Boomin pattern: {metro_pattern['kick'][:8]}... (showing first 8 steps)")
    
    # If you have an audio file to analyze, uncomment this section:
    # print("\n3. Analyzing audio sample...")
    # if os.path.exists('sample.wav'):
    #     analysis = producer.analyze_sample('sample.wav')
    #     print(f"   File: {analysis['file_path']}")
    #     print(f"   Duration: {analysis['duration']:.2f} seconds")
    #     print(f"   Tempo: {analysis['tempo']:.1f} BPM")
    #     print(f"   Predicted Style: {analysis['predicted_style']}")
    #     print(f"   Beat Count: {analysis['beat_count']}")
    
    print("\nKey Features:")
    print("• Audio sampling and analysis")
    print("• Artist style recognition (Travis Scott, Kanye West, Metro Boomin)")
    print("• Beat generation with different patterns")
    print("• Audio processing (time-stretch, pitch-shift)")
    print("• Sample extraction and looping")
    
    print("\nHow to use:")
    print("1. Place audio files in the same directory")
    print("2. Use producer.analyze_sample('your_file.wav') to analyze")
    print("3. Use producer.create_beat('artist_style') to generate beats")
    print("4. Use producer.sample_and_remix() to create remixes")
    
    print("\nNext steps:")
    print("• Add your own audio samples")
    print("• Experiment with different artist styles")
    print("• Train custom style recognition models")
    print("• Integrate with DAW or audio software")
    
    print(f"\nGenerated files:")
    print("• travis_scott_beat.wav")
    print("• kanye_west_beat.wav") 
    print("• metro_boomin_beat.wav")
    
    print("\nAI Producer ready!")

if __name__ == "__main__":
    main()
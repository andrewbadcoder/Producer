#!/usr/bin/env python3
"""
AI-Powered Beat Generation Example
Demonstrates natural language beat creation using Anthropic API
"""

import os
from ai_producer_with_api import EnhancedAIProducer

def main():
    print("AI-Powered Beat Generation Example")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("To use AI-powered beat generation, set your Anthropic API key:")
        print("export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nFalling back to standard beat generation...")
        
        # Use standard producer without AI
        from producer_fixed import AIProducer
        producer = AIProducer()
        
        print("Creating standard beats...")
        styles = ['travis_scott', 'kanye_west', 'metro_boomin']
        for style in styles:
            beat, pattern = producer.create_beat(style)
            filename = f"{style}_standard.wav"
            producer.audio_processor.save_audio(beat, filename)
            print(f"Created: {filename} ({pattern['tempo']} BPM)")
        
        return
    
    # Initialize AI Producer
    try:
        producer = EnhancedAIProducer(api_key=api_key)
        print("AI Producer initialized with Anthropic API!")
    except Exception as e:
        print(f"Error initializing AI Producer: {e}")
        return
    
    # Example prompts to test
    example_prompts = [
        "Make me a Travis Scott type beat that's really dark and atmospheric",
        "I want a Kanye West style beat with soul samples at 110 BPM for 8 bars",
        "Create an aggressive Metro Boomin beat that's perfect for trap",
        "Make a chill beat that sounds like it could be on Astroworld album",
        "Generate a hard-hitting 808 beat at 140 BPM with dark vibes"
    ]
    
    print(f"\nGenerating {len(example_prompts)} AI-powered beats...")
    
    for i, prompt in enumerate(example_prompts, 1):
        print(f"\nPrompt {i}: '{prompt}'")
        
        try:
            # Generate beat from natural language prompt
            audio, pattern, params = producer.create_beat_from_prompt(prompt)
            
            # Save the generated beat
            filename = f"ai_beat_{i}_{params['artist_style']}.wav"
            producer.audio_processor.save_audio(audio, filename)
            
            # Display results
            print(f"   Generated: {filename}")
            print(f"   Artist Style: {params['artist_style']}")
            print(f"   Tempo: {params['tempo']} BPM")
            print(f"   Mood: {params['mood']}")
            print(f"   Duration: {params['duration']} seconds")
            print(f"   Bars: {params['bars']}")
            
            if 'description' in params:
                print(f"   Description: {params['description']}")
            
        except Exception as e:
            print(f"   Error generating beat: {e}")
    
    print(f"\nInteractive Mode:")
    print("Enter your own prompts (type 'quit' to exit):")
    
    beat_counter = len(example_prompts) + 1
    while True:
        try:
            user_prompt = input("\nYour prompt: ").strip()
            
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_prompt:
                continue
            
            print(f"Generating beat from: '{user_prompt}'")
            
            # Generate beat from user prompt
            audio, pattern, params = producer.create_beat_from_prompt(user_prompt)
            
            # Save the beat
            filename = f"user_beat_{beat_counter}_{params['artist_style']}.wav"
            producer.audio_processor.save_audio(audio, filename)
            
            print(f"   Created: {filename}")
            print(f"   Style: {params['artist_style']} | Tempo: {params['tempo']} BPM | Mood: {params['mood']}")
            
            beat_counter += 1
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\nAI-powered beat generation complete!")
    print("Check your directory for generated .wav files")

if __name__ == "__main__":
    main()
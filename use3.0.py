#!/usr/bin/env python3
"""
Ethereal Dreams - A Musical Masterpiece
Composed using the advanced MusicSynth system

This piece demonstrates:
- Complex multi-layered composition
- Advanced humanization and phrasing
- Multiple instruments and effects
- Dynamic tempo changes
- Sophisticated harmonic progressions
"""
import sys

# Import your music maker (adjust the import based on your file structure)
try:
    from XMusic import MusicSynth, Sequencer, FilterType
except ImportError:
    # If the import fails, provide instructions
    print("❌ Could not import MusicSynth, Sequencer, FilterType")
    print("📝 Please ensure this file is in the same directory as your XMusic.py file")
    print("🔧 Or adjust the import statement to match your file structure")
    sys.exit(1)

def create_masterpiece():
    """Create 'Ethereal Dreams' - a cinematic orchestral piece"""
    
    # Initialize the synthesizer with high-quality settings
    synth = MusicSynth(sample_rate=44100)  # Use your default sample rate
    
    # Configure effects that work with your implementation
    try:
        synth.set_reverb(on=True)
        print("🎛️ Reverb enabled")
    except Exception as e:
        print(f"⚠️ Reverb not available: {e}")
    
    try:
        synth.set_chorus(on=True)
        print("🎛️ Chorus enabled")
    except Exception as e:
        print(f"⚠️ Chorus not available: {e}")
    
    # Create the main sequencer
    main_seq = Sequencer(synth, bpm=72, swing=0.1, humanize=True)
    
    # Configure advanced phrasing for emotional expression
    try:
        main_seq.set_phrasing(
            bar_accent=12,
            phrase_crescendo=25, 
            phrase_length=8,
            end_soften=18,
            dynamic_range=40
        )
        print("🎵 Advanced phrasing configured")
    except Exception as e:
        print(f"⚠️ Phrasing not available: {e}")
    
    print("🎼 Composing 'Ethereal Dreams'...")
    print("📝 A cinematic journey through light and shadow")
    
    # === MOVEMENT I: AWAKENING ===
    print("\n🌅 Movement I: Awakening (Gentle introduction)")
    
    # Configure effects if available
    try:
        synth.set_filter(FilterType.LOWPASS, cutoff=8000, resonance=0.3)
        synth.set_delay(time=0.8, feedback=0.4, mix=0.2)
        print("🎛️ Filter and delay configured")
    except Exception as e:
        print(f"⚠️ Some effects not available: {e}")
    
    # Delicate melody in high register
    awakening_melody = [
        ("C5", 1.5, 45), ("E5", 0.5, 50), ("G5", 2.0, 55),
        ("F5", 1.0, 48), ("E5", 1.0, 45), ("D5", 2.0, 42),
        ("C5", 1.5, 40), ("D5", 0.5, 45), ("E5", 4.0, 50)
    ]
    
    # Soft harmonic foundation
    bass_harmony = [
        (["C3", "E3", "G3"], 4.0, 35), (["F3", "A3", "C4"], 4.0, 38),
        (["G3", "B3", "D4"], 4.0, 40), (["C3", "E3", "G3"], 4.0, 33)
    ]
    
    # Add piano melody
    for note, beats, vel in awakening_melody:
        main_seq.add_note(note, beats, vel, channel=0, instrument="piano")
    
    # Add harmonic foundation with electric piano
    for chord, beats, vel in bass_harmony:
        main_seq.add_chord(chord, beats, vel, channel=1, instrument="electric_piano")
    
    main_seq.add_rest(2.0)
    
    # === MOVEMENT II: ASCENSION ===
    print("🚀 Movement II: Ascension (Building energy)")
    
    # Increase tempo and complexity
    try:
        main_seq.add_tempo_change(new_bpm=88, at_beat=main_seq.total_beats)
        print("🎚️ Tempo increased to 88 BPM")
    except Exception as e:
        print(f"⚠️ Tempo changes not available: {e}")
    
    # Configure effects for this movement
    try:
        synth.set_filter(FilterType.BANDPASS, cutoff=2000, resonance=0.6)
        synth.set_distortion(0.1)  # Subtle warmth
        print("🎛️ Bandpass filter and distortion applied")
    except Exception as e:
        print(f"⚠️ Some effects not available: {e}")
    
    # Soaring violin melody
    violin_theme = [
        ("G4", 0.75, 80), ("A4", 0.25, 85), ("B4", 1.0, 90),
        ("C5", 1.5, 95), ("D5", 0.5, 100), ("E5", 2.0, 105),
        ("D5", 1.0, 95), ("C5", 1.0, 90), ("B4", 1.0, 85),
        ("A4", 2.0, 80), ("G4", 2.0, 75)
    ]
    
    # Rich orchestral chords
    string_section = [
        (["C4", "E4", "G4", "C5"], 2.0, 70),
        (["F4", "A4", "C5", "F5"], 2.0, 75),
        (["G4", "B4", "D5", "G5"], 2.0, 80),
        (["E4", "G4", "B4", "E5"], 2.0, 78),
        (["A3", "C4", "E4", "A4"], 2.0, 72),
        (["D4", "F4", "A4", "D5"], 2.0, 76)
    ]
    
    # Layer the violin melody
    for note, beats, vel in violin_theme:
        main_seq.add_note(note, beats, vel, channel=2, instrument="violin")
    
    # Add rich string harmonies
    for chord, beats, vel in string_section:
        main_seq.add_chord(chord, beats, vel, channel=3, instrument="synth_pad")
    
    # === MOVEMENT III: TRANSFORMATION ===
    print("⚡ Movement III: Transformation (Climactic development)")
    
    # Dramatic tempo increase
    try:
        main_seq.add_tempo_change(new_bpm=110, at_beat=main_seq.total_beats)
        print("🎚️ Tempo increased to 110 BPM")
    except Exception as e:
        print(f"⚠️ Tempo changes not available: {e}")
    
    # Add guitar for modern edge
    try:
        synth.set_filter(FilterType.HIGHPASS, cutoff=400, resonance=0.8)
        synth.set_distortion(0.3)
        synth.set_delay(time=0.25, feedback=0.6, mix=0.4)
        print("🎛️ Guitar effects configured")
    except Exception as e:
        print(f"⚠️ Some effects not available: {e}")
    
    # Powerful guitar arpeggios
    guitar_arpeggios = [
        ("E3", 0.25, 110), ("G3", 0.25, 108), ("B3", 0.25, 112), ("E4", 0.25, 115),
        ("G4", 0.25, 118), ("B4", 0.25, 120), ("E5", 0.5, 125), ("B4", 0.5, 115),
        ("F3", 0.25, 110), ("A3", 0.25, 108), ("C4", 0.25, 112), ("F4", 0.25, 115),
        ("A4", 0.25, 118), ("C5", 0.25, 120), ("F5", 0.5, 125), ("C5", 0.5, 115),
        ("G3", 0.25, 112), ("B3", 0.25, 110), ("D4", 0.25, 114), ("G4", 0.25, 117),
        ("B4", 0.25, 120), ("D5", 0.25, 122), ("G5", 1.0, 127), ("D5", 1.0, 118)
    ]
    
    # Thunderous orchestral climax
    orchestral_climax = [
        (["C3", "E3", "G3", "C4", "E4", "G4"], 1.0, 120),
        (["F3", "A3", "C4", "F4", "A4", "C5"], 1.0, 122),
        (["G3", "B3", "D4", "G4", "B4", "D5"], 1.0, 125),
        (["C4", "E4", "G4", "C5", "E5", "G5"], 2.0, 127)
    ]
    
    # Layer guitar arpeggios
    for note, beats, vel in guitar_arpeggios:
        main_seq.add_note(note, beats, vel, channel=4, instrument="electric_guitar")
    
    # Add massive orchestral climax
    for chord, beats, vel in orchestral_climax:
        main_seq.add_chord(chord, beats, vel, channel=5, instrument="synth_pad")
    
    # === MOVEMENT IV: RESOLUTION ===
    print("🕊️ Movement IV: Resolution (Peaceful conclusion)")
    
    # Return to gentle tempo
    try:
        main_seq.add_tempo_change(new_bpm=60, at_beat=main_seq.total_beats)
        print("🎚️ Tempo reduced to 60 BPM for peaceful ending")
    except Exception as e:
        print(f"⚠️ Tempo changes not available: {e}")
    
    # Clear effects for intimate ending
    try:
        synth.set_filter(FilterType.LOWPASS, cutoff=6000, resonance=0.2)
        synth.set_distortion(0.0)
        synth.set_delay(time=1.2, feedback=0.3, mix=0.15)
        print("🎛️ Effects cleared for intimate ending")
    except Exception as e:
        print(f"⚠️ Some effects not available: {e}")
    
    # Gentle piano return to opening theme
    resolution_theme = [
        ("C5", 2.0, 50), ("G4", 2.0, 45), ("E4", 2.0, 48),
        ("C4", 3.0, 40), ("G3", 3.0, 38), ("C3", 6.0, 35)
    ]
    
    # Ethereal pad fadeout
    final_harmony = [
        (["C3", "E3", "G3", "C4"], 6.0, 45),
        (["C3", "E3", "G3"], 6.0, 30),
        (["C3", "G3"], 8.0, 20)
    ]
    
    # Add final piano theme
    for note, beats, vel in resolution_theme:
        main_seq.add_note(note, beats, vel, channel=0, instrument="piano")
    
    # Add ethereal conclusion
    for chord, beats, vel in final_harmony:
        main_seq.add_chord(chord, beats, vel, channel=1, instrument="synth_pad")
    
    # Final silence for reflection
    main_seq.add_rest(4.0)
    
    # === RENDER THE MASTERPIECE ===
    print("\n🎨 Rendering masterpiece...")
    print(f"📊 Total duration: {main_seq.total_beats} beats (~{main_seq.total_beats * 60 / 72:.1f} seconds)")
    print("🎵 Movements: 4")
    print("🎼 Channels used: 6")
    print("🎛️ Effects: Advanced synthesis with your custom FX chain")
    
    # Render with high quality
    try:
        main_seq.render(
            filename="ethereal_dreams_masterpiece.wav",
            midi_filename="ethereal_dreams.mid"
        )
        print("\n✨ 'Ethereal Dreams' has been created!")
        print("🎧 Listen to ethereal_dreams_masterpiece.wav")
        print("🎼 MIDI available at ethereal_dreams.mid")
    except Exception as e:
        print(f"❌ Error during rendering: {e}")
        print("🔧 Please check your MusicSynth implementation")
    
    # Cleanup
    try:
        synth.close()
        print("🧹 Synth cleaned up successfully")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")
    
    print("\n🎭 This piece showcases:")
    print("   • Advanced humanization and phrasing")
    print("   • Multi-instrumental orchestration") 
    print("   • Dynamic tempo changes")
    print("   • Sophisticated effects processing")
    print("   • Emotional musical storytelling")

def create_simple_demo():
    """Create a simpler demo that should work with any MusicSynth setup"""
    
    print("\n" + "="*50)
    print("🎹 Creating Simple Demo: 'Gentle Melody'")
    print("🌸 A basic showcase using core features only")
    
    synth = MusicSynth()
    simple_seq = Sequencer(synth, bpm=90, humanize=True)
    
    # Simple beautiful melody
    melody = [
        ("C4", 1.0, 70), ("E4", 1.0, 75), ("G4", 1.0, 80), ("C5", 2.0, 85),
        ("B4", 1.0, 80), ("A4", 1.0, 75), ("G4", 1.0, 70), ("F4", 2.0, 65),
        ("E4", 1.0, 70), ("D4", 1.0, 75), ("C4", 4.0, 60)
    ]
    
    # Simple chord progression
    chords = [
        (["C3", "E3", "G3"], 4.0, 50), (["F3", "A3", "C4"], 4.0, 55),
        (["G3", "B3", "D4"], 4.0, 50), (["C3", "E3", "G3"], 4.0, 45)
    ]
    
    # Add melody
    for note, beats, vel in melody:
        simple_seq.add_note(note, beats, vel, channel=0, instrument="piano")
    
    # Add accompaniment
    for chord, beats, vel in chords:
        simple_seq.add_chord(chord, beats, vel, channel=1, instrument="electric_piano")
    
    # Render simple demo
    try:
        simple_seq.render(
            filename="gentle_melody_demo.wav",
            midi_filename="gentle_melody_demo.mid"
        )
        print("🎵 'Gentle Melody' demo created successfully!")
    except Exception as e:
        print(f"❌ Error creating demo: {e}")
    
    synth.close()

if __name__ == "__main__":
    try:
        # Create the main masterpiece
        create_masterpiece()
        
    except Exception as e:
        print(f"❌ Error in main masterpiece: {e}")
        print("🔄 Trying simple demo instead...")
        
        # Fallback to simple demo
        create_simple_demo()
    
    print("\n🏆 Composition complete!")
    print("🎵 Your Music Maker has proven its capabilities!")
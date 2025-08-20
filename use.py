# use.py
from music import MusicSynth, Sequencer  # assuming your code is in music_maker.py

def main():
    # Initialize synth
    synth = MusicSynth(sample_rate=44100)
    
    # Initialize sequencer with some swing and humanization
    seq = Sequencer(synth, bpm=100, swing=0.2, humanize=True)

    # Basic demo: C major scale
    notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]
    for n in notes:
        seq.add_note(n, beats=1, velocity=100, channel=0, instrument="piano")

    # Add a rest
    seq.add_rest(beats=2)

    # Add a C major chord
    seq.add_chord(["C4", "E4", "G4"], beats=4, velocity=110, channel=0, instrument="piano")

    # Add something on another instrument for fun
    seq.add_note("C3", beats=2, velocity=120, channel=1, instrument="violin")
    seq.add_note("G3", beats=2, velocity=120, channel=1)

    # Render audio + export MIDI
    seq.render(filename="demo_output.wav", midi_filename="demo_output.mid")

    # Close synth
    synth.close()

if __name__ == "__main__":
    main()

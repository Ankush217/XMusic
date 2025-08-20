from music import MusicSynth, Sequencer
import random

def main():
    synth = MusicSynth(sample_rate=44100)
    seq = Sequencer(synth, bpm=110, swing=0.3, humanize=True)

    # 🎹 Main chord progression (Am – F – C – G)
    chords = [
        ["A3", "C4", "E4"],
        ["F3", "A3", "C4"],
        ["C3", "E3", "G3"],
        ["G3", "B3", "D4"]
    ]
    for c in chords:
        seq.add_chord(c, beats=4, velocity=90, channel=0, instrument="piano")

    # 🎻 Simple violin melody over chords
    melody = ["E4", "G4", "A4", "C5", "B4", "A4", "G4", "E4"]
    for note in melody:
        seq.add_note(note, beats=1, velocity=random.randint(80,110), channel=1, instrument="violin")

    # 🥁 Dumb percussion pattern (hi-hat substitute with piano staccato)
    for i in range(16):
        if i % 2 == 0:
            seq.add_note("C2", beats=0.5, velocity=60, channel=9, instrument="drums")
        else:
            seq.add_rest(beats=0.5)

    # 🎶 Render music
    seq.render(filename="myooceque.wav", midi_filename="myooceque.mid")
    synth.close()

if __name__ == "__main__":
    main()

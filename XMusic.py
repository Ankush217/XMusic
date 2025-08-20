import fluidsynth
import numpy as np
from scipy.io import wavfile
from pathlib import Path
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from typing import List, Union, Dict, Tuple
import random
from collections import deque

from scipy.signal import lfilter, butter
from enum import Enum

_DEFAULT_RATE = 44100

# Generate MIDI note numbers for note names (C0–C8)
NOTES = {}
names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
for octave in range(0, 9):
    for i, name in enumerate(names):
        NOTES[f"{name}{octave}"] = 12 * octave + i

class FilterType(Enum):
    LOWPASS = 1
    HIGHPASS = 2
    BANDPASS = 3

class FXProcessor:
    def __init__(self, sample_rate=_DEFAULT_RATE):
        self.sample_rate = sample_rate
        self.reverb_buffer = np.zeros(sample_rate * 2)  # 2 second buffer
        self.reverb_pos = 0
        self.reverb_decay = 0.5
        self.reverb_mix = 0.3
        self.delay_buffer = np.zeros(sample_rate * 2)  # 2 second buffer
        self.delay_pos = 0
        self.delay_time = 0.3  # seconds
        self.delay_feedback = 0.5
        self.delay_mix = 0.3
        self.distortion_amount = 0.0
        self.filter_type = FilterType.LOWPASS
        self.filter_cutoff = 20000
        self.filter_resonance = 0.5
        self.b, self.a = butter(4, self.filter_cutoff / (sample_rate / 2), btype='lowpass')

    def update_filter(self):
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.filter_cutoff / nyquist
        
        if self.filter_type == FilterType.LOWPASS:
            self.b, self.a = butter(4, normal_cutoff, btype='lowpass')
        elif self.filter_type == FilterType.HIGHPASS:
            self.b, self.a = butter(4, normal_cutoff, btype='highpass')
        elif self.filter_type == FilterType.BANDPASS:
            self.b, self.a = butter(4, [normal_cutoff * 0.8, normal_cutoff * 1.2], btype='bandpass')

    def apply_reverb(self, samples):
        if self.reverb_mix <= 0:
            return samples
        
        wet = np.zeros_like(samples)
        buffer_len = len(self.reverb_buffer)
        
        for i in range(len(samples)):
            pos = (self.reverb_pos + i) % buffer_len
            wet[i] = samples[i] + self.reverb_buffer[pos] * self.reverb_decay
            self.reverb_buffer[pos] = wet[i]
        
        self.reverb_pos = (self.reverb_pos + len(samples)) % buffer_len
        return samples * (1 - self.reverb_mix) + wet * self.reverb_mix

    def apply_delay(self, samples):
        if self.delay_mix <= 0:
            return samples
        
        wet = np.zeros_like(samples)
        delay_samples = int(self.delay_time * self.sample_rate)
        buffer_len = len(self.delay_buffer)
        
        for i in range(len(samples)):
            read_pos = (self.delay_pos - delay_samples + i) % buffer_len
            wet[i] = samples[i] + self.delay_buffer[read_pos] * self.delay_feedback
            self.delay_buffer[(self.delay_pos + i) % buffer_len] = wet[i]
        
        self.delay_pos = (self.delay_pos + len(samples)) % buffer_len
        return samples * (1 - self.delay_mix) + wet * self.delay_mix

    def apply_distortion(self, samples):
        if self.distortion_amount <= 0:
            return samples
        
        return np.tanh(samples * (1 + self.distortion_amount * 10)) * (1 / (1 + self.distortion_amount))

    def apply_filter(self, samples):
        if self.filter_cutoff >= 20000 and self.filter_type == FilterType.LOWPASS:
            return samples
        if self.filter_cutoff <= 20 and self.filter_type == FilterType.HIGHPASS:
            return samples
        
        return lfilter(self.b, self.a, samples)

    def process(self, samples):
        samples = self.apply_filter(samples)
        samples = self.apply_distortion(samples)
        samples = self.apply_reverb(samples)
        samples = self.apply_delay(samples)
        return samples

class DynamicLayer:
    def __init__(self, velocity_range: Tuple[int, int], sample_variations: int = 1):
        self.velocity_min = velocity_range[0]
        self.velocity_max = velocity_range[1]
        self.sample_variations = sample_variations
        self.current_variation = 0
    
    def matches_velocity(self, velocity: int) -> bool:
        return self.velocity_min <= velocity <= self.velocity_max
    
    def get_variation_index(self) -> int:
        if self.sample_variations <= 1:
            return 0
        idx = self.current_variation
        self.current_variation = (self.current_variation + 1) % self.sample_variations
        return idx

class MusicSynth:
    def __init__(self, sample_rate=_DEFAULT_RATE):
        self.sample_rate = sample_rate
        self.fs = fluidsynth.Synth(samplerate=sample_rate)
        self.active_notes = {}  # Track active notes for each channel
        self.reverb_on = False
        self.chorus_on = False
        self.detune_amount = 0.0  # cents
        self.global_volume = 1.0
        self.fx = FXProcessor(sample_rate)
        self.dynamic_layers: Dict[int, List[DynamicLayer]] = {}  # Channel -> layers
        self.round_robin_counters: Dict[Tuple[int, int], int] = {}  # (channel, note) -> counter

        # find SoundFont inside ./data
        sf_path = Path(__file__).parent / "data" / "instruments.sf2"
        if not sf_path.exists():
            raise FileNotFoundError("SoundFont instruments.sf2 not found in ./data/")

        self.sfid = self.fs.sfload(str(sf_path))

        # Default to program 0 if it exists
        try:
            self.fs.program_select(0, self.sfid, 0, 0)
            print("✅ Default instrument: bank=0 preset=0")
        except Exception:
            print("⚠️ Could not select bank=0 preset=0 — try change_instrument() manually.")

        # General MIDI preset map (basic set)
        self.instruments = {
            "piano": (0, 0),
            "bright_piano": (0, 1),
            "electric_piano": (0, 4),
            "guitar_nylon": (0, 24),
            "guitar_steel": (0, 25),
            "electric_guitar": (0, 28),
            "distortion_guitar": (0, 30),
            "violin": (0, 40),
            "trumpet": (0, 56),
            "synth_pad": (0, 88),
            "drums": (128, 0),  # Channel 10 for drums
        }

        # Setup default dynamic layers for channel 0 (piano)
        self.setup_dynamic_layers(0, [
            ((1, 40), 3),   # ppp to pp, 3 variations
            ((41, 80), 4),  # p to f, 4 variations
            ((81, 127), 2)  # ff to fff, 2 variations
        ])

    def setup_dynamic_layers(self, channel: int, layers: List[Tuple[Tuple[int, int], int]]):
        """Configure velocity layers for a channel"""
        self.dynamic_layers[channel] = [DynamicLayer(vrange, variations) for vrange, variations in layers]

    def get_velocity_layer(self, channel: int, velocity: int) -> Tuple[int, int]:
        """Get the appropriate layer and variation index for a velocity"""
        if channel not in self.dynamic_layers:
            return velocity, 0  # No layers configured
        
        for layer in self.dynamic_layers[channel]:
            if layer.matches_velocity(velocity):
                return velocity, layer.get_variation_index()
        
        return velocity, 0  # Fallback

    def change_instrument(self, name="piano", channel=0):
        key = name.lower().replace(" ", "_")
        if key not in self.instruments:
            raise ValueError(f"Instrument '{name}' not found in instrument map")
        bank, preset = self.instruments[key]
        self.fs.program_select(channel, self.sfid, bank, preset)
        print(f"🎹 Changed instrument to {name} (bank={bank}, preset={preset})")

    def set_reverb(self, on=True, room_size=0.7, damping=0.4, width=0.5, level=0.8):
        self.reverb_on = on
        if on:
            self.fs.reverb_on(room_size, damping, width, level)
            self.fx.reverb_mix = level
            self.fx.reverb_decay = damping
        else:
            self.fs.reverb_off()
            self.fx.reverb_mix = 0

    def set_chorus(self, on=True, nr=3, level=2.0, speed=0.3, depth=8.0, type=0):
        self.chorus_on = on
        if on:
            self.fs.chorus_on(nr, level, speed, depth, type)
        else:
            self.fs.chorus_off()

    def set_detune(self, cents=0.0):
        """Set global detuning in cents (-1200 to 1200)"""
        self.detune_amount = max(-1200, min(1200, cents))
        self.fs.pitch_bend(0, int(4096 * (self.detune_amount / 1200)))

    def set_volume(self, volume: float):
        """Set global volume (0.0 to 1.0)"""
        self.global_volume = max(0.0, min(1.0, volume))
        self.fs.volume(0, int(self.global_volume * 127))

    def set_filter(self, filter_type: FilterType, cutoff: float, resonance: float = 0.5):
        """Configure the audio filter"""
        self.fx.filter_type = filter_type
        self.fx.filter_cutoff = max(20, min(20000, cutoff))
        self.fx.filter_resonance = max(0.1, min(1.0, resonance))
        self.fx.update_filter()

    def set_distortion(self, amount: float):
        """Set distortion amount (0.0 to 1.0)"""
        self.fx.distortion_amount = max(0.0, min(1.0, amount))

    def set_delay(self, time: float, feedback: float, mix: float):
        """Configure delay effect"""
        self.fx.delay_time = max(0.01, min(2.0, time))
        self.fx.delay_feedback = max(0.0, min(0.99, feedback))
        self.fx.delay_mix = max(0.0, min(0.5, mix))

    def note_on(self, note: Union[str, int], velocity=100, channel=0):
        """Start a note without generating audio yet"""
        if isinstance(note, str):
            if note not in NOTES:
                raise ValueError(f"Unknown note {note}")
            midi = NOTES[note]
        else:
            midi = note
        
        # Apply velocity layers and round-robin
        velocity, variation = self.get_velocity_layer(channel, velocity)
        
        # Apply slight random detuning for humanization
        if self.detune_amount != 0:
            self.fs.pitch_bend(channel, int(4096 * (random.uniform(-0.5, 0.5) * self.detune_amount / 1200)))
        
        # Apply round-robin variation by slightly altering velocity
        rr_key = (channel, midi)
        if rr_key not in self.round_robin_counters:
            self.round_robin_counters[rr_key] = 0
        variation_velocity = max(1, min(127, velocity + (self.round_robin_counters[rr_key] % 3) - 1))
        self.round_robin_counters[rr_key] += 1
        
        self.fs.noteon(channel, midi, variation_velocity)
        if channel not in self.active_notes:
            self.active_notes[channel] = []
        self.active_notes[channel].append(midi)

    def note_off(self, note: Union[str, int], channel=0):
        """Stop a specific note"""
        if isinstance(note, str):
            if note not in NOTES:
                raise ValueError(f"Unknown note {note}")
            midi = NOTES[note]
        else:
            midi = note
        
        self.fs.noteoff(channel, midi)
        if channel in self.active_notes and midi in self.active_notes[channel]:
            self.active_notes[channel].remove(midi)

    def generate_wave(self, freq, duration, velocity=100, channel=0, note_name=None):
        if note_name is not None:
            midi_note = NOTES[note_name]
        else:
            midi_note = int(round(69 + 12 * np.log2(freq / 440.0)))

        self.note_on(midi_note, velocity=velocity, channel=channel)
        num_samples = int(self.sample_rate * duration)
        samples = self.fs.get_samples(num_samples)
        self.note_off(midi_note, channel=channel)
        
        # Apply FX processing
        samples = self.fx.process(samples)
        
        return np.array(samples, dtype=np.float32)

    def all_notes_off(self, channel=None):
        """Stop all active notes, optionally for a specific channel"""
        if channel is None:
            for ch in list(self.active_notes.keys()):
                for note in self.active_notes[ch]:
                    self.fs.noteoff(ch, note)
                self.active_notes[ch] = []
        elif channel in self.active_notes:
            for note in self.active_notes[channel]:
                self.fs.noteoff(channel, note)
            self.active_notes[channel] = []

    def program_select(self, channel: int, bank: int, preset: int):
        """Select an instrument by bank and preset directly"""
        self.fs.program_select(channel, self.sfid, bank, preset)
        print(f"🎹 Program select: channel={channel}, bank={bank}, preset={preset}")

    def close(self):
        self.all_notes_off()
        self.fs.delete()

class Sequencer:
    def __init__(self, synth: MusicSynth, bpm=120, swing=0.0, humanize=True):
        self.synth = synth
        self.bpm = bpm
        self.swing = max(0.0, min(1.0, swing))  # 0=straight, 1=heavy swing
        self.humanize = humanize
        self.sample_rate = synth.sample_rate
        self.events = []  # holds (type, data...)
        self.tempo_changes = []  # holds (time_in_ticks, new_bpm)
        self.sustain_pedal = False
        self.arpeggio_window = 0.2  # seconds to consider notes as arpeggio
        
        # Articulation rules
        self.articulation_rules = {
            'downbeat_boost': 15,       # Velocity boost for downbeats
            'upbeat_reduce': 10,        # Velocity reduction for upbeats
            'repeated_note_decay': 5,   # Velocity reduction for repeated notes
            'arpeggio_accent': 10       # Accent on top note of arpeggios
        }
        
        # Phrasing rules
        self.phrasing_rules = {
            'bar_accent': 10,           # boost first beat of each bar
            'phrase_crescendo': 20,     # gradually increase velocity over a phrase
            'phrase_length': 4,         # how many bars in a phrase
            'end_soften': 15,           # soften notes at the end of a phrase
            'dynamic_range': 30         # max velocity variation from phrasing
        }
        
        self.last_notes = {}            # Track last played notes per channel
        self.note_history = deque(maxlen=8)  # Track recent notes for arpeggio detection
        self.current_phrase_pos = 0     # Track position in current phrase

    def beats_to_seconds(self, beats: float, current_bpm: float = None) -> float:
        if current_bpm is None:
            current_bpm = self.get_current_bpm_at_beat(0)  # Default to initial BPM
        return 60.0 * beats / current_bpm

    def get_current_bpm_at_beat(self, beat: float) -> float:
        """Get the effective BPM at a specific beat position"""
        if not self.tempo_changes:
            return self.bpm
        
        ticks = int(beat * 480)  # Convert to MIDI ticks
        current_bpm = self.bpm
        for change_ticks, new_bpm in sorted(self.tempo_changes, key=lambda x: x[0]):
            if ticks >= change_ticks:
                current_bpm = new_bpm
            else:
                break
        return current_bpm

    def add_tempo_change(self, new_bpm: float, at_beat: float = 0):
        """Add a tempo change at a specific beat position"""
        ticks = int(at_beat * 480)  # Convert to MIDI ticks
        self.tempo_changes.append((ticks, new_bpm))
        print(f"🎚 Tempo change to {new_bpm} BPM at beat {at_beat}")

    def set_sustain(self, on: bool):
        """Enable/disable sustain pedal effect"""
        self.sustain_pedal = on
        print(f"🎹 Sustain pedal {'ON' if on else 'OFF'}")

    def set_swing(self, amount: float):
        """Set swing amount (0.0 = straight, 1.0 = heavy swing)"""
        self.swing = max(0.0, min(1.0, amount))

    @property
    def total_beats(self):
        total = 0.0
        for event in self.events:
            if event[0] == "note" or event[0] == "chord":
                total += event[2]  # beats
            elif event[0] == "rest":
                total += event[1]  # beats
        return total

    def set_phrasing(self, bar_accent=10, phrase_crescendo=20, phrase_length=4, end_soften=15, dynamic_range=30):
        """Configure musical phrasing rules"""
        self.phrasing_rules = {
            'bar_accent': bar_accent,
            'phrase_crescendo': phrase_crescendo,
            'phrase_length': phrase_length,
            'end_soften': end_soften,
            'dynamic_range': dynamic_range
        }

    def apply_phrasing(self, velocity: int, position_beats: float) -> int:
        """Apply musical phrasing rules to velocity"""
        bar_length = 4  # assuming 4/4 time
        beat_in_bar = int(position_beats) % bar_length
        phrase_len_beats = self.phrasing_rules['phrase_length'] * bar_length
        phrase_pos = position_beats % phrase_len_beats
        
        # Bar accent (downbeat = louder)
        if beat_in_bar == 0:
            velocity += self.phrasing_rules['bar_accent']
        
        # Crescendo over phrase
        crescendo_amount = int((phrase_pos / phrase_len_beats) * self.phrasing_rules['phrase_crescendo'])
        velocity += crescendo_amount
        
        # End of phrase = soften
        if phrase_pos > phrase_len_beats * 0.75:
            velocity -= self.phrasing_rules['end_soften']
        
        # Ensure we stay within dynamic range
        base_velocity = 100  # mid-point
        min_vel = max(1, base_velocity - self.phrasing_rules['dynamic_range']//2)
        max_vel = min(127, base_velocity + self.phrasing_rules['dynamic_range']//2)
        
        return max(min_vel, min(max_vel, velocity))

    def detect_arpeggio(self, notes: List[str]) -> bool:
        """Check if recent notes form an arpeggio pattern"""
        if len(notes) < 3:
            return False
            
        # Check if notes are within arpeggio time window
        if len(self.note_history) > 0:
            last_time = self.note_history[-1][1]
            if (self.current_phrase_pos - last_time) > self.arpeggio_window * (self.bpm/60):
                return False
                
        # Check if notes form a chord (all same beat)
        unique_beats = {beat for _, beat in self.note_history}
        return len(unique_beats) == 1

    def humanize_velocity(self, base_velocity: int, position: float, note: str, channel: int) -> int:
        """Apply humanization rules to velocity"""
        velocity = base_velocity
        
        # Random variation
        if self.humanize:
            velocity += random.randint(-15, 15)
        
        # Apply phrasing rules
        velocity = self.apply_phrasing(velocity, position)
        
        # Downbeat/upbeat rules
        beat_pos = position % 1.0
        if beat_pos < 0.5:  # Downbeat
            velocity += self.articulation_rules['downbeat_boost']
        else:  # Upbeat
            velocity -= self.articulation_rules['upbeat_reduce']
        
        # Repeated note rule
        if channel in self.last_notes and note == self.last_notes[channel]:
            velocity -= self.articulation_rules['repeated_note_decay']
        
        # Arpeggio accent (if detected)
        if isinstance(note, str) and self.detect_arpeggio([note]):
            highest = max([n for n in self.note_history], key=lambda x: NOTES[x[0]])
            if note == highest[0]:
                velocity += self.articulation_rules['arpeggio_accent']
        
        self.last_notes[channel] = note
        self.note_history.append((note, position))
        self.current_phrase_pos = position
        
        return max(1, min(127, velocity))

    def humanize_timing(self, beats: float) -> float:
        """Apply timing humanization"""
        if not self.humanize:
            return beats
        
        # Swing timing for eighth notes
        if self.swing > 0:
            eighth_pos = beats * 2 % 1.0
            if eighth_pos > 0.5:  # Second eighth note gets delayed
                beats += 0.02 * self.swing  # Up to 20ms delay for full swing
        
        # Random timing jitter
        beats += random.uniform(-0.01, 0.01)  # ±10ms jitter
        
        return beats

    def humanize_duration(self, beats: float) -> float:
        """Apply duration humanization"""
        if not self.humanize:
            return beats
        
        # Shorten or lengthen notes randomly
        return beats * random.uniform(0.9, 1.1)

    def add_rest(self, beats=1):
        """Add silence to the sequence"""
        self.events.append(("rest", beats))

    def add_note(self, note, beats=1, velocity=100, channel=0, instrument=None):
        self.events.append(("note", note, beats, velocity, channel, instrument))

    def add_chord(self, notes, beats=2, velocity=100, channel=0, instrument=None):
        self.events.append(("chord", notes, beats, velocity, channel, instrument))

    def render(self, filename="output.wav", midi_filename=None):
        sr = self.sample_rate
        total_seconds = self.total_beats * (60.0 / self.bpm)
        audio_total = np.zeros(int(sr * total_seconds), dtype=np.float32)

        current_pos = 0

        for event in self.events:
            if event[0] == "note":
                _, note, beats, velocity, channel, instr = event
                instr = instr or "piano"
                key = instr.lower().replace(" ", "_")

                if key not in self.synth.instruments:
                    print(f"⚠️ Unknown instrument '{instr}', defaulting to piano")
                    key = "piano"

                bank, preset = self.synth.instruments[key]
                self.synth.program_select(channel, bank, preset)

                freq = 440.0 * (2 ** ((NOTES[note] - 69) / 12.0))
                duration = beats * (60.0 / self.bpm)
                num_samples = int(sr * duration)
                full_audio = self.synth.generate_wave(freq, duration, velocity)

                # Ensure exact length
                if len(full_audio) < num_samples:
                    full_audio = np.pad(full_audio, (0, num_samples - len(full_audio)))
                elif len(full_audio) > num_samples:
                    full_audio = full_audio[:num_samples]

                mix_pos = int(current_pos)
                required_length = mix_pos + num_samples
                if len(audio_total) < required_length:
                    audio_total = np.pad(audio_total, (0, required_length - len(audio_total)))

                audio_total[mix_pos:mix_pos + num_samples] += full_audio
                current_pos += num_samples

            elif event[0] == "chord":
                _, notes, beats, velocity, channel, instr = event
                instr = instr or "piano"
                key = instr.lower().replace(" ", "_")

                if key not in self.synth.instruments:
                    print(f"⚠️ Unknown instrument '{instr}', defaulting to piano")
                    key = "piano"

                bank, preset = self.synth.instruments[key]
                self.synth.program_select(channel, bank, preset)

                duration = beats * (60.0 / self.bpm)
                num_samples = int(sr * duration)
                chord_audio = np.zeros(num_samples, dtype=np.float32)

                for note in notes:
                    freq = 440.0 * (2 ** ((NOTES[note] - 69) / 12.0))
                    note_audio = self.synth.generate_wave(freq, duration, velocity)

                    # Trim or pad to match chord length
                    if len(note_audio) < num_samples:
                        note_audio = np.pad(note_audio, (0, num_samples - len(note_audio)))
                    elif len(note_audio) > num_samples:
                        note_audio = note_audio[:num_samples]

                    chord_audio += note_audio

                mix_pos = int(current_pos)
                required_length = mix_pos + num_samples
                if len(audio_total) < required_length:
                    audio_total = np.pad(audio_total, (0, required_length - len(audio_total)))

                audio_total[mix_pos:mix_pos + num_samples] += chord_audio
                current_pos += num_samples

            elif event[0] == "rest":
                _, beats = event
                current_pos += int(beats * sr)

        # Normalize audio
        audio_total /= np.max(np.abs(audio_total) + 1e-9)

        # Save to file
        wavfile.write(filename, sr, audio_total.astype(np.float32))
        print(f"🎧 Exported audio to {filename}")

        if midi_filename:
            self.export_midi(midi_filename)

    def export_midi(self, filename="output.mid"):
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)

        # Set initial tempo (microseconds per quarter note)
        initial_tempo = mido.bpm2tempo(self.bpm)
        track.append(MetaMessage('set_tempo', tempo=initial_tempo))

        # Add tempo changes if any
        for ticks, new_bpm in self.tempo_changes:
            tempo = mido.bpm2tempo(new_bpm)
            track.append(MetaMessage('set_tempo', tempo=tempo, time=ticks))

        current_time = 0
        current_instruments = {}  # Track instruments per channel

        for event in self.events:
            if event[0] == "note":
                _, note, beats, velocity, channel, instr = event

                # Fallback if instrument is None
                if instr is None:
                    instr = "piano"

                # Change instrument if needed
                if channel not in current_instruments or current_instruments[channel] != instr:
                    key = instr.lower().replace(" ", "_")
                    if key in self.synth.instruments:
                        bank, preset = self.synth.instruments[key]
                        track.append(Message('program_change', channel=channel, program=preset, time=current_time))
                        current_instruments[channel] = instr
                        current_time = 0

                midi_note = NOTES[note]
                ticks = int(beats * 480)  # 480 ticks per quarter note

                # Note on
                track.append(Message('note_on', note=midi_note, velocity=velocity, channel=channel, time=current_time))
                current_time = 0

                # Note off
                track.append(Message('note_off', note=midi_note, velocity=0, channel=channel, time=ticks))

            elif event[0] == "chord":
                _, notes, beats, velocity, channel, instr = event

                if instr is None:
                    instr = "piano"

                if channel not in current_instruments or current_instruments[channel] != instr:
                    key = instr.lower().replace(" ", "_")
                    if key in self.synth.instruments:
                        bank, preset = self.synth.instruments[key]
                        track.append(Message('program_change', channel=channel, program=preset, time=current_time))
                        current_instruments[channel] = instr
                        current_time = 0

                ticks = int(beats * 480)
                midi_notes = [NOTES[n] for n in notes if n in NOTES]

                # Chord on
                for note in midi_notes:
                    track.append(Message('note_on', note=note, velocity=velocity, channel=channel, time=current_time))
                    current_time = 0

                # Chord off
                for note in midi_notes:
                    track.append(Message('note_off', note=note, velocity=0, channel=channel, time=0))
                current_time = ticks

            elif event[0] == "rest":
                _, beats = event
                current_time += int(beats * 480)

        mid.save(filename)
        print(f"🎼 Exported MIDI to {filename}")
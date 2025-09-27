import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from tensorflow.keras.models import load_model

with open("notes.pkl", "rb") as f:
    notes = pickle.load(f)

pitchnames = sorted(set(item for item in notes))
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

n_vocab = len(set(notes))

network_input = []
sequence_length = 100
for i in range(0, len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    network_input.append([note_to_int[char] for char in seq_in])

start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]

model = load_model("music_model.h5")

prediction_output = []

for note_index in range(200):
    prediction_input = np.reshape(pattern, (1, len(pattern), 1))
    prediction_input = prediction_input / float(n_vocab)

    prediction = model.predict(prediction_input, verbose=0)
    index = np.argmax(prediction)
    result = pitchnames[index]
    prediction_output.append(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]

# Convert output notes to MIDI
offset = 0
output_notes = []

for pattern in prediction_output:
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes_in_chord = [note.Note(int(n)) for n in notes_in_chord]
        for n in notes_in_chord:
            n.storedInstrument = instrument.Piano()
        new_chord = chord.Chord(notes_in_chord)
        new_chord.offset = offset
        output_notes.append(new_chord)
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')

print("🎵 Music generated! Saved as generated_music.mid")

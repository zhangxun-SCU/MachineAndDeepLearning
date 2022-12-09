import music21

s=music21.converter.parse('music/1.mid')
# 获取持续的时间每个音符
print([note.duration.quarterLength for note in s.flat.notesAndRests])

for note in s.flat.notesAndRests:
    if isinstance(note, music21.note.Rest):
        print("r")
    elif isinstance(note,music21.note.Note):
        print(note.name,note.octave,note.pitch,note.pitch.midi,note.duration.quarterLength)
    # 取和弦
    else:
        for c_note in note.notes:
            print(c_note.name, c_note.pitch.midi,c_note.duration.quarterLength)
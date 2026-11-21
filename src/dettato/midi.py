"""Write the pipeline's tracks as a Standard MIDI File at 120 BPM."""

from struct import pack


TICKS_PER_BEAT = 220
TICKS_PER_SECOND = 2 * TICKS_PER_BEAT
PITCHED_CHANNELS = tuple(channel for channel in range(16) if channel != 9)


def _variable_length(value):
    encoded = [value & 0x7F]
    while value > 0x7F:
        value >>= 7
        encoded.append((value & 0x7F) | 0x80)
    return bytes(reversed(encoded))


def _write_track(stream, events):
    data = bytearray()
    previous_tick = 0
    for tick, message in events:
        data.extend(_variable_length(tick - previous_tick))
        data.extend(message)
        previous_tick = tick
    data.extend(b"\x01\xff\x2f\x00")
    stream.write(pack(">4sI", b"MTrk", len(data)))
    stream.write(data)


def write_midi(stream, tracks):
    stream.write(pack(">4sIHHH", b"MThd", 6, 1, len(tracks) + 1, TICKS_PER_BEAT))
    _write_track(stream, [(0, b"\xff\x58\x04\x04\x02\x18\x08"), (0, b"\xff\x51\x03\x07\xa1\x20")])
    for index, track in enumerate(tracks):
        channel = 9 if track.is_drum else PITCHED_CHANNELS[index % len(PITCHED_CHANNELS)]
        program = 0 if track.is_drum else int(track.program)
        name = track.name.encode("latin-1", errors="replace")
        events = [(0, b"\xff\x03" + _variable_length(len(name)) + name), (0, bytes([0xC0 | channel, program]))]
        notes = []
        for on, off, pitch, velocity in track.notes:
            start = round(float(on) * TICKS_PER_SECOND)
            end = max(start + 1, round(float(off) * TICKS_PER_SECOND))
            notes.append((start, bytes([0x90 | channel, int(pitch), int(velocity)])))
            notes.append((end, bytes([0x90 | channel, int(pitch), 0])))
        _write_track(stream, events + sorted(notes))

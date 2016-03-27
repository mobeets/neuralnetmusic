import midi
import numpy as np

# r=(21, 109)
NUM_KEYS = 88
infile = '/Users/mobeets/code/neuralnetmusic/midifiles/train/ashover_simple_chords_1.mid'

def get_empty_roll(pattern):
    maxlen = 0
    maxptch = 0
    for track in pattern:
        track.make_ticks_abs()
        lens = [event.tick for event in track if isinstance(event, midi.NoteEvent)]
        if lens:
            maxlen = max(maxlen, max(lens))
        # ptch = [event.get_pitch() for event in track if isinstance(event, midi.NoteEvent)]
        # if ptch:
        #     maxptch = max(maxptch, max(ptch))
    # print maxlen, maxptch
    return np.zeros((maxlen+1, NUM_KEYS))

def midiread(infile):
    pattern = midi.read_midifile(infile)
    # pattern = [pattern[1]] # just one track
    roll = get_empty_roll(pattern)
    for track in pattern:
        track.make_ticks_abs()
        for event in track:
            if isinstance(event, midi.NoteEvent) and event.get_velocity() > 0:
                # print event.tick, event.length, event.get_velocity(), event.get_pitch()
                roll[event.tick, event.get_pitch()] = 1
    return roll

## write file

def midiwrite(roll, outfile, note_length=20, resolution=220, vel=90, pitch_offset=0):
    ticks, ptchs = np.nonzero(roll)
    
    # have to duplicate items at later times, to turn them off
    all_ticks = np.hstack((ticks, ticks + note_length))
    all_ptchs = np.hstack((ptchs, ptchs))
    is_note_on = np.zeros(all_ticks.shape)
    is_note_on[:ticks.shape[0]] = 1
    ix = np.argsort(all_ticks) # must enter in chronological order, whether on or off

    pattern = midi.Pattern(resolution=resolution)
    track = midi.Track()
    pattern.append(track)

    last_tick = 0
    for tick, ptch, do_on in zip(all_ticks[ix], all_ptchs[ix], is_note_on[ix]):
        played_last = tick-1 >= 0 and roll[tick-1,ptch] == 1
        played_next = tick+1 < roll.shape[0] and roll[tick+1,ptch] == 1

        ptch += pitch_offset
        if do_on and (not played_last or tick%4 == 0):
            on = midi.NoteOnEvent(tick=(tick-last_tick), velocity=vel, pitch=ptch)
            track.append(on)
            last_tick = tick
        elif not do_on and (not played_next or tick%4 == 3):
            off = midi.NoteOffEvent(tick=(tick-last_tick), pitch=ptch)
            track.append(off)
            last_tick = tick

    print pattern

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile(outfile, pattern)

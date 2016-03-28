import numpy as np
import midi
# import midiparser2

# r=(21, 109)
NUM_KEYS = 88

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

def midiread_tracks(infile, match_name_fcn=None):
    if match_name_fcn is None:
        match_name_fcn = lambda x: True
    pattern = midi.read_midifile(infile)
    for track in pattern:
        for event in track:
            if isinstance(event, midi.TrackNameEvent) and match_name_fcn(event.text):
                print event.text
    
def midiread_sniff(infile):
    pattern = midi.read_midifile(infile)
    print pattern.resolution
    # print pattern[1]
    for track in pattern:
        track.make_ticks_abs()
        for event in track:
            if isinstance(event, midi.TrackNameEvent):
                print event.text
            if isinstance(event, midi.ProgramChangeEvent) or isinstance(event, midi.ProgramNameEvent):
                print event

def midiread(infile, dt=1, desired_resolution=None, match_name=None, sum_rolls=True):
    """
    warning: dt > 1 might lead to missing notes,
        which might in turn lead to never turning notes off
    """
    pattern = midi.read_midifile(infile)
    if desired_resolution is not None:
        dt = pattern.resolution/desired_resolution
    rolls = get_empty_roll(pattern) if sum_rolls else []
    c = 0
    for track in pattern:
        # track must have meta text to be included
        keep_track = False if match_name else True
        roll = get_empty_roll(pattern)
        track.make_ticks_abs()
        for event in track:
            if isinstance(event, midi.NoteEvent):
                # note onset is either a NoteOnEvent with zero velocity, or a NoteOffEvent
                if event.get_velocity() > 0 and not isinstance(event, midi.NoteOffEvent):
                    roll[int(event.tick/dt), event.get_pitch()] = 1
                # note offset
                else:
                    note_off = int(event.tick/dt)
                    if roll[note_off, event.get_pitch()] == 1:
                        continue
                    note_on = np.where(roll[:note_off, event.get_pitch()] == 1)[0][-1]
                    roll[note_on:note_off+1, event.get_pitch()] = 1
            elif match_name and isinstance(event, midi.TrackNameEvent):
                if match_name.lower() == event.text.lower():
                    keep_track = True
        if keep_track:
            if sum_rolls:
                rolls += roll
            else:
                rolls.append(roll)
    return rolls

# def midiread2(infile):
#     notes = midiparser2.process_notes(infile) # columns: (onset time, pitch, duration, velocity)

## write file

def midiwrite(roll, outfile, resolution=220, vel=127, pitch_offset=0, patch_num=None):
    ticks, ptchs = np.nonzero(roll)
    ix = np.argsort(ticks) # must enter in chronological order, whether on or off

    c = 0
    pattern = midi.Pattern(resolution=resolution)
    track = midi.Track()
    pattern.append(track)
    if patch_num is not None:
        patch = midi.ProgramChangeEvent(tick=0, channel=0, data=[patch_num])
        track.append(patch)

    last_tick = 0
    for tick, ptch in zip(ticks[ix], ptchs[ix]):
        played_last = tick-1 >= 0 and roll[tick-1,ptch] == 1
        played_next = tick+1 < roll.shape[0] and roll[tick+1,ptch] == 1

        ptch += pitch_offset
        if not played_last:# or tick%4 == 0:
            on = midi.NoteOnEvent(tick=(tick-last_tick), velocity=vel, pitch=ptch)
            track.append(on)
            last_tick = tick
        if not played_next:# or tick%4 == 3:
            off = midi.NoteOffEvent(tick=(tick-last_tick), pitch=ptch)
            track.append(off)
            last_tick = tick

    # print pattern

    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)
    midi.write_midifile(outfile, pattern)

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
    track_names = []
    pattern = midi.read_midifile(infile)
    for track in pattern:
        for event in track:
            if isinstance(event, midi.TrackNameEvent) and match_name_fcn(event.text):
                track_names.append(event.text)
    print '\n'.join(track_names)
    return track_names
    
def midiread_sniff(infile):
    pattern = midi.read_midifile(infile)
    print pattern.resolution
    for track in pattern:
        track.make_ticks_abs()
        for event in track:
            if isinstance(event, midi.TrackNameEvent):
                print event.text
            if isinstance(event, midi.ProgramChangeEvent) or isinstance(event, midi.ProgramNameEvent):
                print event

def midiread(infile, dt=1, desired_resolution=None, match_name_fcn=None, sum_rolls=True):
    """
    warning: dt > 1 might lead to missing notes,
        which might in turn lead to never turning notes off
    """
    if match_name_fcn is None:
        match_name_fcn = lambda x: True
        match_name = False
    else:
        match_name = True
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
                if event.get_pitch() >= NUM_KEYS or event.get_pitch() < 0:
                    print "WARNING: found pitch out of range: " + str(event.get_pitch())
                    continue
                if event.get_velocity() > 0 and not isinstance(event, midi.NoteOffEvent):
                    roll[int(event.tick/dt), event.get_pitch()] = 1
                # note offset
                else:
                    note_off = int(event.tick/dt)
                    if roll[note_off, event.get_pitch()] == 1:
                        continue
                    note_ons = np.where(roll[:note_off, event.get_pitch()] == 1)[0]
                    if len(note_ons) == 0:
                        print "WARNING: No previous note onset found."
                        continue
                    note_on = note_ons[-1]
                    roll[note_on:note_off+1, event.get_pitch()] = 1
            elif match_name and isinstance(event, midi.TrackNameEvent):
                if match_name_fcn(event.text):
                    keep_track = True
        if keep_track:
            if sum_rolls:
                rolls += roll
            else:
                rolls.append(roll)
    return rolls

## write file
def midiwrite(rolls, outfile, resolution=220, vel=127, pitch_offset=0, patch_nums=None):
    """
    rolls [nticks npitches ntracks]: binary piano roll
    """
    if rolls.ndim == 2:
        rolls = np.resize(rolls, rolls.shape + (1,))
    if patch_nums is None:
        patch_nums = []
    try: # convert single digit to list, so there's one patch_num per roll
        patch_nums = [int(patch_nums)]*rolls.shape[2]
    except TypeError:
        pass
    assert len(patch_nums) == rolls.shape[2] or len(patch_nums) == 0
    pattern = midi.Pattern(resolution=resolution)
    for ir in xrange(rolls.shape[2]):
        roll = rolls[:,:,ir]
        track = midi.Track()
        if patch_nums:
            patch = midi.ProgramChangeEvent(tick=0, channel=ir, data=[patch_nums[ir]])
            track.append(patch)

        ticks, ptchs = np.nonzero(roll)
        ix = np.argsort(ticks) # must enter in chronological order, whether on or off
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
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        pattern.append(track)
    midi.write_midifile(outfile, pattern)

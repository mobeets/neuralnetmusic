import os.path
import glob
import cPickle
import numpy as np
import midiparser
import DBN

RESOLUTION = 2
TRAIN_DIR = 'midifiles/train'
NUM_KEYS = 88

def blockwise_view(a, blockshape, require_aligned_blocks=True):
    """
    source: https://github.com/ilastik/lazyflow/blob/master/lazyflow/utility/blockwise_view.py
    
    >>> a = numpy.arange(1,21).reshape(4,5)
    >>> print a
    [[ 1  2  3  4  5]
     [ 6  7  8  9 10]
     [11 12 13 14 15]
     [16 17 18 19 20]]
    >>> view = blockwise_view(a, (2,2), False)
    >>> print view
    [[[[ 1  2]
       [ 6  7]]

      [[ 3  4]
       [ 8  9]]]

     [[[11 12]
       [16 17]]

      [[13 14]
       [18 19]]]]
   """
    assert a.flags['C_CONTIGUOUS'], "This function relies on the memory layout of the array."
    blockshape = tuple(blockshape)
    view_shape = tuple(np.array(a.shape) / blockshape) + blockshape
    if require_aligned_blocks:
        assert (np.mod(a.shape, blockshape) == 0).all(), \
            "blockshape {} must divide evenly into array shape {}"\
            .format( blockshape, a.shape )
    intra_block_strides = a.strides
    inter_block_strides = tuple(a.strides * np.array(blockshape))
    return np.lib.stride_tricks.as_strided(a, 
                                              shape=view_shape, 
                                              strides=(inter_block_strides+intra_block_strides))

def make_chunked_training_data(train_dir=TRAIN_DIR, max_len=64, resolution=RESOLUTION, inter_outdir=None):
    """
    returns np.array with shape (nrows, NUM_KEYS*max_len)
        where each row is an num_keys-by-pitch matrix respresenting 4 bars of a midi file
    """
    train_files = glob.glob(os.path.join(train_dir, '*.mid'))
    ds = []
    c = 0
    for infile in train_files:
        m = midiparser.midiread(infile, desired_resolution=resolution, sum_rolls=True)
        s = blockwise_view(m, (max_len, NUM_KEYS), require_aligned_blocks=False)[:,0]
        # print infile, m.shape, s.shape
        if inter_outdir is not None:
            for i in xrange(s.shape[0]):
                outfile = os.path.join(inter_outdir, str(i + c) + '-.mid')
                midiparser.midiwrite(s[i,:,:], outfile, resolution=resolution, patch_nums=82)
        c += s.shape[0]
        ds.append(s)
    ds = np.vstack(ds)
    ds = np.swapaxes(ds, 1, 2) # (nrows, 88, max_len)
    ds1 = np.reshape(ds, (ds.shape[0], NUM_KEYS*max_len)) # (nrows, 88*max_len)
    assert np.all(ds1[1,:].reshape([88, 64]) == ds[1,:])
    return ds1

def make_chunked_combined_training_data(indirs=[TRAIN_DIR], max_len=64, resolution=RESOLUTION, inter_outdirs=None):
    """
    returns np.array with shape (nrows, NUM_KEYS*max_len)
        where each row is an num_keys-by-pitch matrix respresenting 4 bars of a midi file
    """
    allsamelen = lambda items: len(set([len(x) for x in items])) == 1
    allsameshape = lambda items: len(set([x.shape for x in items])) == 1

    # load file names for each indir
    train_files = {}
    ds = {}
    cs = []
    for i,indir in enumerate(indirs):
        train_files[i] = glob.glob(os.path.join(indir, '*.mid'))
        ds[i] = []
        cs.append(0)
    assert allsamelen(train_files.values())

    for i in xrange(len(train_files[0])):
        print 'Reading file {0}'.format(i)

        # load current midifile in each indir
        ms = []
        for j in xrange(len(indirs)):
            infile = train_files[j][i]
            ms.append(midiparser.midiread(infile, desired_resolution=resolution, sum_rolls=True))
        maxticks = max([m.shape[0] for m in ms])

        # extend length of each midifile so that all are the same
        mfs = []
        for j, m in enumerate(ms):
            nzers = maxticks - m.shape[0]
            m = np.lib.pad(m, ((0, nzers), (0,0)), 'constant', constant_values=(0,))
            mfs.append(m)
        assert allsameshape(mfs)

        for j, m in enumerate(mfs):
            s = blockwise_view(m, (max_len, NUM_KEYS), require_aligned_blocks=False)[:,0]
            if inter_outdirs is not None:
                if not os.path.exists(inter_outdirs[j]):
                    os.mkdir(inter_outdirs[j])
                for k in xrange(s.shape[0]):
                    outfile = os.path.join(inter_outdirs[j], str(k + cs[j]) + '.mid')
                    midiparser.midiwrite(s[k,:,:], outfile, resolution=resolution, patch_nums=82)
            cs[j] += s.shape[0]
            ds[j].append(s)
    dsout = []
    for j in xrange(len(indirs)):
        d = ds[j]
        d = np.vstack(d)
        d = np.swapaxes(d, 1, 2) # (nrows, 88, max_len)
        d1 = np.reshape(d, (d.shape[0], NUM_KEYS*max_len)) # (nrows, 88*max_len)
        assert np.all(d1[1,:].reshape([88, 64]) == d[1,:])
        dsout.append(d)
    return dsout

def make_pruned_training_data(pretrain_dir, outdir, track_fcn):
    infiles = glob.glob(os.path.join(pretrain_dir, '*.mid'))
    for infile in infiles:
        outfile = infile.replace(pretrain_dir, outdir).replace('.mid', '_processed.mid')
        print infile, outfile
        roll = midiparser.midiread(infile, desired_resolution=2, match_name_fcn=track_fcn, sum_rolls=True)
        midiparser.midiwrite(roll, outfile, resolution=2, vel=127, pitch_offset=0, patch_nums=82)
        print '-----'

def track_list(train_dir=TRAIN_DIR, match_name_fcn=None):
    train_files = glob.glob(os.path.abspath(train_dir))
    for infile in train_files:
        print infile
        midiparser.midiread_tracks(infile, match_name_fcn)
        print '------------'

def reconstruct(mdlfile='output/metallica_gtr-model.pickle', datfile='input/metallica_gtr-data.pickle', ind=None):
    # load data and model
    dbn = DBN.load_from_dump(mdlfile)
    raw_x = cPickle.load(open(datfile, 'rb')).astype(dtype=DBN.NUMPY_DTYPE)

    # choose input data
    if ind is not None:
        inp = raw_x[ind,:].reshape(1, raw_x.shape[1])

        # find latents, then sample from latents to reconstruct
        out = dbn.latents(inp)
        outs = np.tile(out, (10,1))
        out_p = dbn.sample(outs, threshold=0.0)

        # write input and its reconstruction
        outfile = 'output/test_inp.midi'
        midiparser.midiwrite(inp.reshape(88, 64).T, outfile, resolution=2, patch_nums=82)
        outfile = 'output/test_out.midi'
        midiparser.midiwrite(out_p.T, outfile, resolution=2, patch_nums=82)

    else:
        latents = []
        for ind in xrange(raw_x.shape[0]):
            inp = raw_x[ind,:].reshape(1, raw_x.shape[1])
            out = dbn.latents(inp)
            latents.append(out)
        return latents

def write_training(obj, outfile):
    with open(outfile, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

def load_training(infile):
    return cPickle.load(open(infile)).astype(dtype=np.float64)

def combine_training(train_files, outfile):
    " warning: only works if all pieces are the same size "
    objs = []
    for train_file in train_files:
        obj = load_training(train_file)
        print obj.shape
        objs.append(obj)
    obj = np.hstack(objs)
    print obj.shape
    write_training(obj, outfile)

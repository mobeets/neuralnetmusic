import os.path
import glob
import numpy as np
import midiparser

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

def make_training_data(train_dir=TRAIN_DIR, max_len=64, resolution=RESOLUTION):
    """
    returns np.array with shape (nrows, NUM_KEYS*max_len)
        where each row is an num_keys-by-pitch matrix respresenting 4 bars of a midi file
    """
    train_files = glob.glob(os.path.join(train_dir, '*.mid'))
    ds = []
    for infile in train_files:
        m = midiparser.midiread(infile, desired_resolution=resolution)
        s = blockwise_view(m, (max_len, NUM_KEYS), require_aligned_blocks=False)[:,0]
        print s.shape
        ds.append(s)
    ds = np.vstack(ds)
    ds = np.swapaxes(ds, 1, 2) # (nrows, 88, max_len)
    ds1 = np.reshape(ds, (ds.shape[0], NUM_KEYS*max_len)) # (nrows, 88*max_len)
    assert np.all(ds1[1,:].reshape([88, 64]) == ds[1,:])
    return ds1

def save_training_data(pretrain_dir, outdir):
    infiles = glob.glob(os.path.join(pretrain_dir, '*.mid'))
    match_name_fcn = lambda x: any(y in x.lower() for y in ['guitar', 'gtr'])
    for infile in infiles:
        outfile = infile.replace(pretrain_dir, outdir).replace('.mid', '_processed.mid')
        print infile
        print outfile
        print '-----'
        roll = midiparser.midiread(infile, desired_resolution=2, match_name_fcn=match_name_fcn, sum_rolls=True)
        midiparser.midiwrite(roll, outfile, resolution=2, vel=127, pitch_offset=0, patch_num=82)

def track_list(train_dir=TRAIN_DIR, match_name_fcn=None):
    train_files = glob.glob(os.path.abspath(train_dir))
    for infile in train_files:
        print infile
        midiparser.midiread_tracks(infile, match_name_fcn)
        print '------------'

if __name__ == '__main__':
    pretrain_dir = 'midifiles/metallica'
    outdir = 'midifiles/metallica_gtr'
    outfile = 'midifiles/metallica_gtr-data.pickle'
    # save_training_data(pretrain_dir, outdir)
    my_obj = make_training_data(train_dir=outdir, max_len=64)
    
    import cPickle
    print cPickle.load(open('./joplin-data.pickle', 'rb')).astype(dtype=np.float64).shape
    print my_obj.shape
    with open(outfile, 'wb') as f:
        cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print cPickle.load(open(outfile)).astype(dtype=np.float64).shape

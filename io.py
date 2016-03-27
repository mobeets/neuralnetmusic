import numpy as np
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

def make_training_data(train_dir='midifiles/train/*.mid', max_len=64, resolution=RESOLUTION):
    """
    returns np.array with shape (nrows, NUM_KEYS*max_len)
        where each row is an num_keys-by-pitch matrix respresenting 4 bars of a midi file
    """
    train_files = glob.glob(train_dir)[:10]
    ds = []
    for infile in train_files:
        m = midiparser.midiread(infile, desired_resolution=resolution)
        s = blockwise_view(m, (max_len, NUM_KEYS), require_aligned_blocks=False)[:,0]
        ds.append(s)
    ds = np.vstack(ds)
    ds = np.swapaxes(ds, 1, 2) # (nrows, 88, max_len)
    ds1 = np.reshape(ds, (ds.shape[0], NUM_KEYS*max_len)) # (nrows, 88*max_len)
    assert np.all(ds1[1,:].reshape([88, 64]) == ds[1,:])
    return ds1

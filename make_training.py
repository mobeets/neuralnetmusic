import glob
import os.path
import numpy as np
import cPickle
import midiparser
import dataio

gtr_fcn = lambda x: any(y in x.lower() for y in ['guitar', 'gtr']) and 'bass' not in x.lower()
bass_fcn = lambda x: any(y in x.lower() for y in ['bass'])

def make_training(pretrain_dir, outdir, track_fcn):
    # outpiecedir = outdir + '-pieces'
    # outfile = outdir + '.pickle'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # if not os.path.exists(outpiecedir):
    #     os.mkdir(outpiecedir)
    dataio.make_pruned_training_data(pretrain_dir, outdir, track_fcn)

    # obj = dataio.make_chunked_training_data(train_dir=outdir, max_len=64, inter_outdir=outpiecedir)
    # dataio.write_training(obj, outfile)
    # return obj

def combine_training(pretrain_dir, indirs, track_fcns, outdir):

    print "PRUNING"
    for i, indir in enumerate(indirs):
        dataio.make_pruned_training_data(pretrain_dir, indir, track_fcns[i])

    print "CHUNKING"
    objs = dataio.make_chunked_combined_training_data(indirs=indirs, inter_outdirs=[indir + '-pieces' for indir in indirs])
    for i, obj in enumerate(objs):
        outfile = indirs[i] + '.pickle'
        dataio.write_training(obj, outfile)

    print "WRITING SOLOS AND COMBINED"
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outfile = outdir + '.pickle'
    infiles = [indir + '.pickle' for indir in indirs]
    dataio.combine_training(infiles, outfile)

if __name__ == '__main__':
    pretrain_dir = 'input/metallica'
    gtrdir = 'input/metallica_gtr0'
    bassdir = 'input/metallica_bass'
    outdir = 'input/metallica_gtr_bass'
    combine_training(pretrain_dir, [gtrdir, bassdir], [gtr_fcn, bass_fcn], outdir)

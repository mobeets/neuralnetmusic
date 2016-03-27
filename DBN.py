"""
"""
import cPickle
import os
import sys
import time
import os.path as path
import copy

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from DeepLearningTutorials.code.mlp import HiddenLayer
from DeepLearningTutorials.code.rbm import RBM

from PIL import Image

import myparser
from _midi.utils import midiwrite
from midiparser import midiwrite as midiwrite2

# compute_test_value is 'off' by default, meaning this feature is inactive
theano.config.compute_test_value = 'off' # Use 'warn' to activate this feature

# For switching between 32 and 64 bit systems, because Theano is a little silly
# like that.
NUMPY_DTYPE = numpy.float64

# start-snippet-1
class AutoencodingDBN(object):
    """
    An autoencoding Deep Belief Network, based on the classifying DBN in the
    Theano tutorial.  (Most of the code is copied over.)
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[1000, 1000, 1000]):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.layer_sizes = hidden_layers_sizes

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.x_mask = T.matrix('x_mask')    # For partial information.

        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)


            # Construct an RBM that shared weights with this layer
            rbm_layer = RBM(numpy_rng=numpy_rng,
                            theano_rng=theano_rng,
                            input=layer_input,
                            n_visible=input_size,
                            n_hidden=hidden_layers_sizes[i],
                            W=sigmoid_layer.W,
                            hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)
            self.params.extend(rbm_layer.params)

        # And build the upside-down network.  This shares parameters with the
        # forward network. Except the weights are transposed and stuff.

        # The "isolated" layers let you run only the upside-down part of the
        # network, for generation.  The non-isolated layers are connected to
        # the forward, compressing part of the network, and are used for
        # training.
        reverse_input = self.sigmoid_layers[-1].output
        self.isolated_reverse_input = theano.shared(
            numpy.zeros([10, hidden_layers_sizes[-1]]))
        isolated_input = self.isolated_reverse_input
        self.reverse_layers = [None] * self.n_layers
        self.isolated_reverse = [None] * self.n_layers
        for i in reversed(xrange(self.n_layers)):    
            if i == 0:
                out_size = n_ins
            else:
                out_size = hidden_layers_sizes[i-1]
            reverse_sigmoid = HiddenLayer(rng=numpy_rng,
                input=reverse_input,
                n_in=hidden_layers_sizes[i],
                n_out=out_size,
                W=self.sigmoid_layers[i].W.T,
                b=self.rbm_layers[i].vbias,
                activation=T.nnet.sigmoid
            )
            isolated_sigmoid = HiddenLayer(rng=numpy_rng,
                input=isolated_input,
                n_in=hidden_layers_sizes[i],
                n_out=out_size,
                W=self.sigmoid_layers[i].W.T,
                b=self.rbm_layers[i].vbias,
                activation=T.nnet.sigmoid
            )
            
            reverse_input = reverse_sigmoid.output
            isolated_input = isolated_sigmoid.output
            self.reverse_layers[i] = reverse_sigmoid
            self.isolated_reverse[i] = isolated_sigmoid


        # The fine-tune cost is the reconstruction error of the entire net.
        self.finetune_cost = ((self.x - self.reverse_layers[0].output)**2).sum()

        # The cost for training the generative net - in this case, self.x is
        # completely disconnected, and we feed a pattern into the reverse net.
        self.generative_cost = ((self.x - self.isolated_reverse[0].output)**2).sum()

        # The l1 cost is for generating constrained samples of the input.  (Aka
        # harmonizing a melody.)  Given a melody in self.x and a mask
        # self.x_mask of which parts of self.x actually matter, it computes the
        # error between the generated sample and the melody.
        self.l1_cost = (((self.x - self.isolated_reverse[0].output) * self.x_mask)**2).sum()

    def dump_params(self, outLoc):
        """
        Takes all of the weights, and stores them as numpy arrays.
        This is so the params are portable between GPU machines and CPU machines.
        To load the params, you need to call load_from_dump, which re-makes your
        DBN.
        """
        dump = {}
        for layer in range(self.n_layers):
            dump[(layer, 0)] = numpy.array(self.sigmoid_layers[layer].W.get_value())
            dump[(layer, 1)] = numpy.array(self.sigmoid_layers[layer].b.get_value())
            dump[(layer, 2)] = numpy.array(self.reverse_layers[layer].b.get_value())
        cPickle.dump(dump, open(outLoc, 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # number of batches
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, train_set_x, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''

        index = T.lscalar('index')  # index to a [mini]batch
        n_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
            }
        )

        test_score_i = theano.function(
            [index],
            self.finetune_cost,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
            }
        )

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_batches)]

        return train_fn, test_score


    def build_generative_finetune_fns(self, train_set_outputs, train_set_labels,
                                      batch_size, learning_rate):
        index = T.lscalar('index')  # index to a [mini]batch
        n_batches = train_set_outputs.get_value(borrow=True).shape[0] / batch_size

        # compute the gradients with respect to the model parameters
        # First, only one of the RBM biases is actually a parameter of the
        # generative model, so we have to fix that.
        gen_params = []
        for i in range(self.n_layers):
            gen_params.append(self.rbm_layers[i].vbias)
            gen_params.append(self.rbm_layers[i].W)
        gparams = T.grad(self.generative_cost, gen_params)

        # compute list of fine-tuning updates
        updates = []
        for param, gparam in zip(gen_params, gparams):
            updates.append((param, param - gparam * learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.generative_cost,
            updates=updates,
            givens={
                self.x: train_set_outputs[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.isolated_reverse[-1].input: train_set_labels[
                    index * batch_size: (index + 1) * batch_size
                ],
            }
        )

        test_score_i = theano.function(
            [index],
            self.generative_cost,
            givens={
                self.x: train_set_outputs[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.isolated_reverse[-1].input: train_set_labels[
                    index * batch_size: (index + 1) * batch_size
                ],
            }
        )

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_batches)]

        return train_fn, test_score

    def generate(self, top_level):
        """
        Make a new piano roll, given top level values.  (Uses the backwards
        section of the network to make a sample.)
        """
        generator = theano.function(
            [],
            self.reverse_layers[0].output,
            givens={
                self.reverse_layers[-1].input: top_level
            }
        )
        return generator()

    def label(self, to_label, x_mask, learning_rate):
        """
        Estimate top layer, given an incomplete layer 1.
        x_mask represents which values of to_label are unknown.
        """
        grad = T.grad(self.l1_cost, self.isolated_reverse_input)
        # compute list of fine-tuning updates
        updates = (self.isolated_reverse_input, 
            self.isolated_reverse_input - grad * learning_rate)

        train_fn = theano.function(
            inputs=[],
            outputs=self.l1_cost,
            updates=[updates],
            givens={
                self.x: to_label,
                self.x_mask: x_mask,
            }
        )
        return train_fn

    def train_dbn(self, data_file, finetune_lr=0.01, pretraining_epochs=100,
        pretrain_lr=0.05, k=1, training_epochs=1000, batch_size=10):

        raw_x = cPickle.load(open(data_file, 'rb')).astype(dtype=NUMPY_DTYPE)
        train_set_x = theano.shared(raw_x)
        

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
        print n_train_batches

        # start-snippet-2
        #########################
        # PRETRAINING THE MODEL #
        #########################
        print '... getting the pretraining functions'
        pretraining_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size,
                                                    k=k)


        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        for i in xrange(self.n_layers - 1):
            # go through pretraining epochs
            for epoch in xrange(pretraining_epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                                                lr=pretrain_lr))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print numpy.mean(c)

        end_time = time.clock()
        # end-snippet-2
        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))

        # If you'd like to try out different parameters for the fine-tuner only,
        # you can cache the initial model state, so you don't have to pre-train
        # every time.
        cPickle.dump(self, open('initial-model.pickle', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
        ########################
        # FINETUNING THE MODEL #
        ########################

        # get the training, validation and testing function for the model

        print '... getting the finetuning functions'
        use_autoencoder = False
        if use_autoencoder:
            train_fn, test_model = self.build_finetune_functions(
                train_set_x=train_set_x,
                batch_size=batch_size,
                learning_rate=finetune_lr
            )
        else:
            raw_labels = numpy.random.randint(2,
                size=[raw_x.shape[0], self.layer_sizes[-1]])\
                .astype(dtype=numpy.float64)
            labels = theano.shared(raw_labels)
            train_fn, test_model = self.build_generative_finetune_fns(
                train_set_outputs=train_set_x,
                train_set_labels=labels,
                batch_size=batch_size,
                learning_rate=finetune_lr 
            )

        print '... finetuning the model'
        # early-stopping parameters
        patience = 4 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.    # wait this much longer when a new best is
                                  # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatches before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0

        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:

                    validation_losses = test_model()
                    this_validation_loss = numpy.mean(validation_losses)
                    print(
                        'epoch %i, minibatch %i/%i, validation error %f %%'
                        % (
                            epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if (
                            this_validation_loss < best_validation_loss *
                            improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f, '
                'obtained at iteration %i, '
            ) % (best_validation_loss, best_iter + 1)
        )
        print >> sys.stderr, ('The fine tuning code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time)
                                                  / 60.))
        self.dump_params('./my-model.pickle')

    def sample(self, top_level=None, rootLoc='./output', save=True, threshold=0.5,
            filename='test.midi'):
        """
        Generates a sample from the trained neural net.  top_level is a 10 x
        [size of top layer] matrix whose rows contain values for the top
        layer.  Most of the time, I only use the first row, but you can only
        process data in increments of batch_size.
        """
        if top_level is None:
            top_level_size = self.layer_sizes[-1]
            # top_level = numpy.random.randint(2, size=[10, top_level_size])\
            #     .astype(dtype=NUMPY_DTYPE)
            top_level = numpy.random.rand(10, top_level_size)\
                .astype(dtype=NUMPY_DTYPE)
        output = self.generate(top_level)
        output = output.reshape([10, 88*64])
        firstIm = output[0, :].reshape([88, 64])
        # Makes a little picture of the piano roll.
        outIm = Image.fromarray((firstIm*255).astype('uint8'))
        outIm.save(path.join(rootLoc, 'test.png'))
        if threshold is not None:
            firstIm[firstIm > threshold] = 1
            firstIm[firstIm <= threshold] = 0
        if save:
            midiwrite(path.join(rootLoc, filename), firstIm.T, r=(12, 109), dt=64)
            print firstIm.T.shape
            outfile = path.join(rootLoc, filename.replace('.mid', '_2.mid'))
            midiwrite2(firstIm.T, outfile, pitch_offset=12, resolution=2, note_length=1)
        return firstIm

    def label_from_file(self, rootLoc, fileLoc, learn_rate, n_iters, threshold):
        """
        Given a xml file at fileLoc, harmonizes the melody in the xml file, by
        doing gradient descent on the top hidden layer of the network.  This
        gives us an estimate of the top layer activations that might generate
        the melody. We then run the network forwards to get the entire harmony
        from the top level activations that we estimate.
        """
        noteReader = myparser.LegatoNoteAdder(64)
        myparser.read(fileLoc, noteReader.handle)
        snippet = noteReader.mtx
        mask = melody_blocker(snippet)

        linear_snippet = snippet.reshape([88*64])
        linear_mask = mask.reshape([88*64])
        in_data = numpy.zeros([10, 88*64])
        x_mask = numpy.zeros([10, 88*64])
        for i in range(10):
            in_data[i, :] = linear_snippet
            x_mask[i, :] = linear_mask


        # Do gradient descent to estimate the activations on layer 1.
        new_vals = theano.shared(
            value=numpy.random.sample([10, self.layer_sizes[-1]]),
        )
        f = theano.function(
            inputs=[],
            updates=[(self.isolated_reverse_input, new_vals)],
        )
        f()
        trainer = self.label(in_data, x_mask, learn_rate)
        for i in range(n_iters):
            print trainer()

        # Then, generate using it.
        result = dbn.sample(self.isolated_reverse_input, rootLoc=rootLoc, save=False,
            threshold=threshold)
        # Add the melody back onto the snippet.
        final = result * (1.0 - mask)
        final = final + snippet
        final[final > 0.5] = 1
        midiwrite(path.join(rootLoc, 'test.midi'), final.T, r=(12, 109), dt=64)
        return final

def musicxml_to_midi(infile):
    noteReader = myparser.LegatoNoteAdder(64)
    myparser.read(infile, noteReader.handle)
    snippet = noteReader.mtx
    midiwrite(infile.replace('.xml', '.midi'), snippet.T, r=(12, 109), dt=64)

def melody_blocker(snippet):
    """
    Makes a mask where anything above the top line of the snippet is 1.  Also
    enforces empty space a major 2nd above and below the melody.  (This means
    the optimizer will consider any note above the top line of the melody, or
    too close to the melody, wrong.)
    """
    envelope = numpy.copy(snippet)
    _, length = snippet.shape
    for i in range(length):
        occupied = [x for x in range(88) if snippet[x, i] != 0]
        if len(occupied) == 0:
            continue
        top = max(occupied)
        envelope[top:, i] = 1
        for pitch in occupied:
            envelope[pitch-2:pitch+3, i] = 1
    return envelope

def load_from_dump(inLoc):
    """
    Loads data from dumped state (generated by dumped_params), and creates a
    new DBN.
    """
    dump = cPickle.load(open(inLoc, 'rb'))
    # Get the number of layers.
    max_layer = 0
    for layer, _ in dump:
        if layer > max_layer:
            max_layer = layer
    max_layer += 1
    # Get the size of each layer.
    layer_sizes = []
    for layer in range(max_layer):
        layer_sizes.append(len(dump[(layer, 1)]))
    # For now, the size of the input is fixed at 88x64, but you can read that
    # out of the dump, as well.
    dbn = AutoencodingDBN(numpy_rng=numpy.random.RandomState(),
        n_ins=88*64,
        hidden_layers_sizes=layer_sizes)
    for layer in range(max_layer):
        dbn.sigmoid_layers[layer].W.set_value(dump[(layer, 0)])
        dbn.sigmoid_layers[layer].b.set_value(dump[(layer, 1)])
        dbn.reverse_layers[layer].b.set_value(dump[(layer, 2)])
    return dbn

if __name__ == '__main__':
    if sys.argv[1] == 'train':
        dbn = AutoencodingDBN(numpy_rng=numpy.random.RandomState(),
            n_ins=88*64,
            hidden_layers_sizes=[1024, 256, 64])
        dbn.train_dbn('./joplin-data.pickle')
        exit()
    dbn = load_from_dump('./joplin-model.pickle')
    import sys
    if sys.argv[1] == 'sample':
        dbn.sample(threshold=0.5)
    elif sys.argv[1] == 'harmonize': 
        dbn.label_from_file('output', sys.argv[2],
            0.01, 500, 0.4)
    elif sys.argv[1] == 'xmltomidi':
        musicxml_to_midi(sys.argv[2])
    else:
        print "invalid command"

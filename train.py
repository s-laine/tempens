#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of NVIDIA Corporation nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

###################################################################################################
# Helper class for forking stdout/stderr into a file.
###################################################################################################

class Tap:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.file = None
        pass

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert(self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None

###################################################################################################
# Global init.
###################################################################################################

import os, sys
stdout_tap = Tap(sys.stdout)
stderr_tap = Tap(sys.stderr)
sys.stdout = stdout_tap
sys.stderr = stderr_tap

import config, warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % config.cuda_device_number
os.environ['THEANO_FLAGS'] = config.theano_flags
warnings.filterwarnings('ignore', message = "downsample module has been moved to the pool module")
    
import report, thread_utils, time
import numpy as np
np.random.seed(config.random_seed)
    
print "CUDA_VISIBLE_DEVICES=" + os.environ['CUDA_VISIBLE_DEVICES']
print "THEANO_FLAGS=" + os.environ['THEANO_FLAGS']
import theano
from theano import tensor as T
import lasagne
import scipy
from collections import OrderedDict 
import json, math
import theano_utils
import pickle
sys.setrecursionlimit(10000)

###################################################################################################
# Image save function that deals correctly with channels.
###################################################################################################

def save_image(filename, img):
    if len(img.shape) == 3:
        if img.shape[0] == 1:            
            img = img[0] # CHW -> HW (saves as grayscale)
        else:            
            img = np.transpose(img, (1, 2, 0)) # CHW -> HWC (as expected by toimage)

    scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(filename)

###################################################################################################
# Dataset loaders.
###################################################################################################

def load_cifar_10():
    import cPickle
    def load_cifar_batches(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join(config.data_dir, 'cifar-10', fn), 'rb') as f:
                data = cPickle.load(f)
            images.append(np.asarray(data['data'], dtype='float32').reshape(-1, 3, 32, 32) / np.float32(255))
            labels.append(np.asarray(data['labels'], dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_cifar_batches(['data_batch_%d' % i for i in (1, 2, 3, 4, 5)])
    X_test, y_test = load_cifar_batches('test_batch')

    return X_train, y_train, X_test, y_test

def load_cifar_100():
    import cPickle
    def load_cifar_file(fn):
        with open(os.path.join(config.data_dir, 'cifar-100', fn), 'rb') as f:
            data = cPickle.load(f)
        images = np.asarray(data['data'], dtype='float32').reshape(-1, 3, 32, 32) / np.float32(255)
        labels = np.asarray(data['fine_labels'], dtype='int32')
        return images, labels

    X_train, y_train = load_cifar_file('train')
    X_test, y_test = load_cifar_file('test')

    return X_train, y_train, X_test, y_test

def load_svhn():
    import cPickle
    def load_svhn_files(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            with open(os.path.join(config.data_dir, 'svhn', fn), 'rb') as f:
                X, y = cPickle.load(f)
            images.append(np.asarray(X, dtype='float32') / np.float32(255))
            labels.append(np.asarray(y, dtype='int32'))
        return np.concatenate(images), np.concatenate(labels)

    X_train, y_train = load_svhn_files(['train_%d.pkl' % i for i in (1, 2, 3)])
    X_test, y_test = load_svhn_files('test.pkl')

    return X_train, y_train, X_test, y_test

def load_tinyimages(indices, output_array=None, output_start_index=0):
    images = output_array
    if images is None:
        images = np.zeros((len(indices), 3, 32, 32), dtype='float32')
    assert(images.shape[0] >= len(indices) + output_start_index and images.shape[1:] == (3, 32, 32))
    with open(os.path.join(config.data_dir, 'tinyimages', 'tiny_images.bin'), 'rb') as f:
        for i, idx in enumerate(indices):
            f.seek(3072 * idx)
            images[output_start_index + i] = np.fromfile(f, dtype='uint8', count=3072).reshape(3, 32, 32).transpose((0, 2, 1)) / np.float32(255)
    return images

def whiten_norm(x):
    x = x - np.mean(x, axis=(1, 2, 3), keepdims=True)
    x = x / (np.mean(x ** 2, axis=(1, 2, 3), keepdims=True) ** 0.5)
    return x

def prepare_dataset(result_subdir, X_train, y_train, X_test, y_test, num_classes):

    # Whiten input data.

    if config.whiten_inputs == 'norm':
        X_train = whiten_norm(X_train)
        X_test = whiten_norm(X_test)
    elif config.whiten_inputs == 'zca':
        whitener = ZCA(x=X_train)
        X_train = whitener.apply(X_train)
        X_test = whitener.apply(X_test)
    elif config.whiten_inputs is not None:
        print("Unknown input whitening mode '%s'." % config.whiten_inputs)
        exit()

    # Pad according to the amount of jitter we plan to have.

    p = config.augment_translation 
    if p > 0:
        X_train = np.pad(X_train, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')
        X_test = np.pad(X_test, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')

    # Random shuffle.

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Corrupt some of the labels if needed.

    num_labels = len(y_train) if config.num_labels == 'all' else config.num_labels
    if config.corruption_percentage > 0:
        corrupt_labels = int(0.01 * num_labels * config.corruption_percentage)
        corrupt_labels = min(corrupt_labels, num_labels)
        print("Corrupting %d labels." % corrupt_labels)
        for i in range(corrupt_labels):
            y_train[i] = np.random.randint(0, num_classes)
    
    # Reshuffle.

    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Construct mask_train. It has a zero where label is unknown, and one where label is known.

    if config.num_labels == 'all':
        # All labels are used.
        mask_train = np.ones(len(y_train), dtype=np.float32)
        print("Keeping all labels.")
    else:
        # Assign labels to a subset of inputs.
        num_img = min(num_classes, 20)
        max_count = config.num_labels // num_classes
        print("Keeping %d labels per class." % max_count)
        img_count = min(max_count, 32)
        label_image = np.zeros((X_train.shape[1], 32 * num_img, 32 * img_count))
        mask_train = np.zeros(len(y_train), dtype=np.float32)
        count = [0] * num_classes
        for i in range(len(y_train)):
            label = y_train[i]
            if count[label] < max_count:
                mask_train[i] = 1.0
                if count[label] < img_count and label < num_img:
                    label_image[:, label * 32 : (label + 1) * 32, count[label] * 32 : (count[label] + 1) * 32] = X_train[i, :, p:p+32, p:p+32]
            count[label] += 1

        # Dump out some of the labeled digits.
        save_image(os.path.join(result_subdir, 'labeled_inputs.png'), label_image)

    # Draw in auxiliary data from the tiny images dataset.

    if config.aux_tinyimg is not None:
        print("Augmenting with unlabeled data from tiny images dataset.")
        with open(os.path.join(config.data_dir, 'tinyimages', 'tiny_index.pkl'), 'rb') as f:
            tinyimg_index = pickle.load(f)

        if config.aux_tinyimg == 'c100':
            print("Using all classes common with CIFAR-100.")

            with open(os.path.join(config.data_dir, 'cifar-100', 'meta'), 'rb') as f:
                cifar_labels = pickle.load(f)['fine_label_names']
            cifar_to_tinyimg = { 'maple_tree': 'maple', 'aquarium_fish' : 'fish' }
            cifar_labels = [l if l not in cifar_to_tinyimg else cifar_to_tinyimg[l] for l in cifar_labels]

            load_indices = sum([list(range(*tinyimg_index[label])) for label in cifar_labels], [])
        else:
            print("Using %d random images." % config.aux_tinyimg)

            num_all_images = max(e for s, e in tinyimg_index.values())
            load_indices = np.arange(num_all_images)
            np.random.shuffle(load_indices)
            load_indices = load_indices[:config.aux_tinyimg]
            load_indices.sort() # Some coherence in seeks.

        # Load the images.

        num_aux_images = len(load_indices)
        print("Loading %d auxiliary unlabeled images." % num_aux_images)
        Z_train = load_tinyimages(load_indices)

        # Whiten and pad.

        if config.whiten_inputs == 'norm':
            Z_train = whiten_norm(Z_train)
        elif config.whiten_inputs == 'zca':
            Z_train = whitener.apply(Z_train)
        Z_train = np.pad(Z_train, ((0, 0), (0, 0), (p, p), (p, p)), 'reflect')

        # Concatenate to training data and append zeros to labels and mask.
        X_train = np.concatenate((X_train, Z_train))
        y_train = np.concatenate((y_train, np.zeros(num_aux_images, dtype='int32')))
        mask_train = np.concatenate((mask_train, np.zeros(num_aux_images, dtype='float32')))

    # Zero out masked-out labels for maximum paranoia.
    for i in range(len(y_train)):
        if mask_train[i] != 1.0:
            y_train[i] = 0

    return X_train, y_train, mask_train, X_test, y_test

###################################################################################################
# Network I/O.
###################################################################################################

def load_network(filename):
    print("Importing network from '%s'." % filename)
    with open(filename, 'rb') as f:
        net = pickle.load(f)

    stack = [net]
    il = None
    while len(stack) > 0:
        il = stack.pop()
        if hasattr(il, 'input_layer'):
            stack.append(il.input_layer)
        elif hasattr(il, 'input_layers'):
            stack += il.input_layers
        else:
           break

    input_var = il.input_var
    return net, input_var

def save_network(net, filename):
    print ("Exporting network to '%s' .." % filename),
    with open(filename, 'wb') as f:
        pickle.dump(net, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")

###################################################################################################
# Network construction.
###################################################################################################

from lasagne.layers import InputLayer, ReshapeLayer, FlattenLayer, Upscale2DLayer, MaxPool2DLayer, DropoutLayer, ConcatLayer, DenseLayer, NINLayer
from lasagne.layers import GaussianNoiseLayer, Conv2DLayer, Pool2DLayer, GlobalPoolLayer, NonlinearityLayer, FeaturePoolLayer, DimshuffleLayer, ElemwiseSumLayer
from lasagne.utils import floatX
from zca_bn import ZCA
from zca_bn import mean_only_bn as WN

def build_network(input_var, num_input_channels, num_classes):
    conv_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'filter_size': (3, 3),
        'stride': (1, 1),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    nin_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    dense_defs = {
        'W': lasagne.init.HeNormal(1.0),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.softmax
    }

    wn_defs = {
        'momentum': config.batch_normalization_momentum
    }

    net = InputLayer        (     name='input',    shape=(None, num_input_channels, 32, 32), input_var=input_var)
    net = GaussianNoiseLayer(net, name='noise',    sigma=config.augment_noise_stddev)
    net = WN(Conv2DLayer    (net, name='conv1a',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv1b',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv1c',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = MaxPool2DLayer    (net, name='pool1',    pool_size=(2, 2))
    net = DropoutLayer      (net, name='drop1',    p=.5)
    net = WN(Conv2DLayer    (net, name='conv2a',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv2b',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(Conv2DLayer    (net, name='conv2c',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = MaxPool2DLayer    (net, name='pool2',    pool_size=(2, 2))
    net = DropoutLayer      (net, name='drop2',    p=.5)
    net = WN(Conv2DLayer    (net, name='conv3a',   num_filters=512, pad=0,      **conv_defs), **wn_defs)
    net = WN(NINLayer       (net, name='conv3b',   num_units=256,               **nin_defs),  **wn_defs)
    net = WN(NINLayer       (net, name='conv3c',   num_units=128,               **nin_defs),  **wn_defs)
    net = GlobalPoolLayer   (net, name='pool3')    
    net = WN(DenseLayer     (net, name='dense',    num_units=num_classes,       **dense_defs), **wn_defs)

    return net

###################################################################################################
# Training utils.
###################################################################################################

def rampup(epoch):
    if epoch < config.rampup_length:
        p = max(0.0, float(epoch)) / float(config.rampup_length)
        p = 1.0 - p
        return math.exp(-p*p*5.0)
    else:
        return 1.0

def rampdown(epoch):
    if epoch >= (config.num_epochs - config.rampdown_length):
        ep = (epoch - (config.num_epochs - config.rampdown_length)) * 0.5
        return math.exp(-(ep * ep) / config.rampdown_length)
    else:
        return 1.0

def robust_adam(loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0e-8):
    # Convert NaNs to zeros.
    def clear_nan(x):
        return T.switch(T.isnan(x), np.float32(0.0), x)

    new = OrderedDict()
    pg = zip(params, lasagne.updates.get_or_compute_grads(loss, params))
    t = theano.shared(lasagne.utils.floatX(0.))

    new[t] = t + 1.0 
    coef = learning_rate * T.sqrt(1.0 - beta2**new[t]) / (1.0 - beta1**new[t])
    for p, g in pg:
        value = p.get_value(borrow=True)
        m = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
        v = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
        new[m] = clear_nan(beta1 * m + (1.0 - beta1) * g)
        new[v] = clear_nan(beta2 * v + (1.0 - beta2) * g**2)
        new[p] = clear_nan(p - coef * new[m] / (T.sqrt(new[v]) + epsilon))

    return new

###################################################################################################
# Training iterators.
###################################################################################################

def iterate_minibatches(inputs, targets, batch_size):
    assert len(inputs) == len(targets)
    num = len(inputs)
    indices = np.arange(num) 
    crop = config.augment_translation
    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            excerpt = indices[start_idx : start_idx + batch_size]
            yield len(excerpt), inputs[excerpt, :, crop:crop+32, crop:crop+32], targets[excerpt]

def iterate_minibatches_augment_pi(inputs, labels, mask, batch_size):
    assert len(inputs) == len(labels) == len(mask)
    crop = config.augment_translation
    
    num = len(inputs)
    if config.max_unlabeled_per_epoch is None:
        indices = np.arange(num)
    else:        
        labeled_indices   = [i for i in range(num) if mask[i] > 0.0]
        unlabeled_indices = [i for i in range(num) if mask[i] == 0.0]
        np.random.shuffle(unlabeled_indices)
        indices = labeled_indices + unlabeled_indices[:config.max_unlabeled_per_epoch] # Limit the number of unlabeled inputs per epoch.
        indices = np.asarray(indices)
        num = len(indices)

    np.random.shuffle(indices)

    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            excerpt = indices[start_idx : start_idx + batch_size]
            noisy_a, noisy_b = [], []
            for img in inputs[excerpt]:
                if config.augment_mirror and np.random.uniform() > 0.5:
                    img = img[:, :, ::-1]
                t = config.augment_translation
                ofs0 = np.random.randint(-t, t + 1) + crop
                ofs1 = np.random.randint(-t, t + 1) + crop
                img_a = img[:, ofs0:ofs0+32, ofs1:ofs1+32]
                ofs0 = np.random.randint(-t, t + 1) + crop
                ofs1 = np.random.randint(-t, t + 1) + crop
                img_b = img[:, ofs0:ofs0+32, ofs1:ofs1+32]
                noisy_a.append(img_a)
                noisy_b.append(img_b)
            yield len(excerpt), excerpt, noisy_a, noisy_b, labels[excerpt], mask[excerpt]

def iterate_minibatches_augment_tempens(inputs, labels, mask, targets, batch_size):
    assert len(inputs) == len(labels) == len(mask) == len(targets)
    crop = config.augment_translation

    num = len(inputs)
    if config.max_unlabeled_per_epoch is None:
        indices = np.arange(num)
    else:        
        labeled_indices   = [i for i in range(num) if mask[i] > 0.0]
        unlabeled_indices = [i for i in range(num) if mask[i] == 0.0]
        np.random.shuffle(unlabeled_indices)
        indices = labeled_indices + unlabeled_indices[:config.max_unlabeled_per_epoch] # Limit the number of unlabeled inputs per epoch.
        indices = np.asarray(indices)
        num = len(indices)

    np.random.shuffle(indices)

    for start_idx in range(0, num, batch_size):
        if start_idx + batch_size <= num:
            excerpt = indices[start_idx : start_idx + batch_size]
            noisy = []
            for img in inputs[excerpt]:
                if config.augment_mirror and np.random.uniform() > 0.5:
                    img = img[:, :, ::-1]
                t = config.augment_translation
                ofs0 = np.random.randint(-t, t + 1) + crop
                ofs1 = np.random.randint(-t, t + 1) + crop
                img = img[:, ofs0:ofs0+32, ofs1:ofs1+32]
                noisy.append(img)
            yield len(excerpt), excerpt, noisy, labels[excerpt], mask[excerpt], targets[excerpt]

###################################################################################################
# Main training function.
###################################################################################################

def run_training(monitor_filename=None):

    # Sanity check network type.

    if config.network_type not in ['pi', 'tempens']:
        print("Unknown network type '%s'." % config.network_type)
        exit()

    # Create the result directory and basic run data.

    result_subdir = report.create_result_subdir(config.result_dir, config.run_desc)
    print "Saving results to", result_subdir

    # Start dumping stdout and stderr into result directory.

    stdout_tap.set_file(open(os.path.join(result_subdir, 'stdout.txt'), 'wt'))
    stderr_tap.set_file(open(os.path.join(result_subdir, 'stderr.txt'), 'wt'))

    # Set window title if on Windows.

    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleA('%s - Gpu %d' % (os.path.split(result_subdir)[1], config.cuda_device_number))
    except:
        pass

    # Export run information.

    report.export_sources(os.path.join(result_subdir, 'src'))
    report.export_run_details(os.path.join(result_subdir, 'run.txt'))
    report.export_config(os.path.join(result_subdir, 'config.txt'))

    # Load the dataset.

    print("Loading dataset '%s'..." % config.dataset)

    if config.dataset == 'cifar-10':
        X_train, y_train, X_test, y_test = load_cifar_10()
    elif config.dataset == 'cifar-100':
        X_train, y_train, X_test, y_test = load_cifar_100()
    elif config.dataset == 'svhn':
        X_train, y_train, X_test, y_test = load_svhn()
    else:
        print("Unknown dataset '%s'." % config.dataset)
        exit()

    # Calculate number of classes.

    num_classes = len(set(y_train))
    assert(set(y_train) == set(y_test) == set(range(num_classes))) # Check that all labels are in range [0, num_classes-1]
    print("Found %d classes in training set, %d in test set." % (len(set(y_train)), len(set(y_test))))

    # Prepare dataset and print stats.

    X_train, y_train, mask_train, X_test, y_test = prepare_dataset(result_subdir, X_train, y_train, X_test, y_test, num_classes)
    print("Got %d training inputs, out of which %d are labeled." % (len(X_train), sum(mask_train)))
    print("Got %d test inputs." % len(X_test))

    #----------------------------------------------------------------------------
    # Prepare to train.
    #----------------------------------------------------------------------------

    print("Network type is '%s'." % config.network_type)

    # Prepare Theano variables for inputs and targets

    input_var = T.tensor4('inputs')
    label_var = T.ivector('labels')
    learning_rate_var = T.scalar('learning_rate')
    adam_beta1_var = T.scalar('adam_beta1')
    input_vars = [input_var]

    scaled_unsup_weight_max = config.unsup_weight_max
    if config.num_labels != 'all':
        scaled_unsup_weight_max *= 1.0 * config.num_labels / X_train.shape[0]

    if config.network_type == 'pi':
        input_b_var = T.tensor4('inputs_b')
        mask_var = T.vector('mask')
        unsup_weight_var = T.scalar('unsup_weight')
        input_vars.append(input_b_var)
    elif config.network_type == 'tempens':
        mask_var = T.vector('mask')
        target_var = T.matrix('targets')
        unsup_weight_var = T.scalar('unsup_weight')
    
    # Load/create the network.

    if config.load_network_filename is not None:
        net, input_var = load_network(config.load_network_filename)
        input_vars = [input_var]
        if config.network_type == 'pi':
            input_vars.append(input_b_var)
    else:
        print("Building network and compiling functions...")
        net = build_network(input_var, X_train.shape[1], num_classes)

    # Export topology report.

    with open(os.path.join(result_subdir, 'network-topology.txt'), 'wt') as fout:
        for line in report.generate_network_topology_info(net):
            print(line)
            fout.write(line + '\n')

    # Initialization updates and function.

    lasagne.layers.get_output(net, init=True)
    init_updates = [u for l in lasagne.layers.get_all_layers(net) for u in getattr(l, 'init_updates', [])]
    init_fn = theano.function(input_vars, [], updates=init_updates, on_unused_input='ignore')

    # Get training predictions, BN updates.

    train_prediction = lasagne.layers.get_output(net)
    if config.network_type == 'pi':
        train_prediction_b = lasagne.layers.get_output(net, inputs=input_b_var) # Second branch.
    bn_updates = [u for l in lasagne.layers.get_all_layers(net) for u in getattr(l, 'bn_updates', [])]

    # Training loss.

    if config.network_type == 'pi':
        train_loss = T.mean(lasagne.objectives.categorical_crossentropy(train_prediction, label_var) * mask_var, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
        train_loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(train_prediction, train_prediction_b), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
    elif config.network_type == 'tempens':
        train_loss = T.mean(lasagne.objectives.categorical_crossentropy(train_prediction, label_var) * mask_var, dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
        train_loss += unsup_weight_var * T.mean(lasagne.objectives.squared_error(train_prediction, target_var), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)

    # ADAM update expressions for training. 

    params = lasagne.layers.get_all_params(net, trainable=True)
    updates = robust_adam(train_loss, params, learning_rate=learning_rate_var, beta1=adam_beta1_var, beta2=config.adam_beta2, epsilon=config.adam_epsilon).items()

    # Training function.

    if config.network_type == 'pi':
        train_fn = theano_utils.function([input_var, input_b_var, label_var, mask_var, learning_rate_var, adam_beta1_var, unsup_weight_var], [train_loss], updates=updates+bn_updates, on_unused_input='warn')
    elif config.network_type == 'tempens':
        train_fn = theano_utils.function([input_var, label_var, mask_var, target_var, learning_rate_var, adam_beta1_var, unsup_weight_var], [train_loss, train_prediction], updates=updates+bn_updates, on_unused_input='warn')

    # Validation prediction, loss, and accuracy.

    test_prediction = lasagne.layers.get_output(net, deterministic=True)
    test_loss = T.mean(lasagne.objectives.categorical_crossentropy(test_prediction, label_var), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), label_var), dtype=theano.config.floatX, acc_dtype=theano.config.floatX)

    # Validation function.

    val_fn = theano_utils.function([input_var, label_var], [test_loss, test_acc], on_unused_input='warn')

    #----------------------------------------------------------------------------
    # Start training.
    #----------------------------------------------------------------------------

    print("Starting training.")

    if config.max_unlabeled_per_epoch is not None:
        print("Limiting number of unlabeled inputs per epoch to %d." % config.max_unlabeled_per_epoch)

    training_csv = report.GenericCSV(os.path.join(result_subdir, 'training.csv'),
        'Epoch', 'EpochTime', 'TrainLoss', 'TestLoss', 'TestAccuracy', 'LearningRate')

    # Initial training variables for temporal ensembling.

    if config.network_type == 'tempens':
        ensemble_prediction = np.zeros((len(X_train), num_classes))
        training_targets = np.zeros((len(X_train), num_classes))

    #----------------------------------------------------------------------------
    # Training loop.
    #----------------------------------------------------------------------------

    for epoch in range(config.start_epoch, config.num_epochs):

        # Export network snapshot every 50 epochs.

        if (epoch % 50) == 0 and epoch != config.start_epoch:
            save_network(net, os.path.join(result_subdir, 'network-snapshot-%03d.pkl' % epoch))

        # Evaluate up/down ramps.

        rampup_value = rampup(epoch)
        rampdown_value = rampdown(epoch)

        # Initialize WN/MOBN layers with a properly augmented minibatch.

        if epoch == 0:
            if config.network_type == 'pi':
                minibatches = iterate_minibatches_augment_pi(X_train, np.zeros((len(X_train),)), np.zeros((len(X_train),)), config.minibatch_size)
                for (n, indices, inputs_a, inputs_b, labels, mask) in minibatches: 
                    init_fn(inputs_a, inputs_b)
                    break
            elif config.network_type == 'tempens':
                minibatches = iterate_minibatches_augment_tempens(X_train, np.zeros((len(X_train),)), np.zeros((len(X_train),)), np.zeros((len(X_train),)), config.minibatch_size)
                for (n, indices, inputs, labels, mask, targets) in minibatches: 
                    init_fn(inputs)
                    break

        # Initialize epoch predictions for temporal ensembling.

        if config.network_type == 'tempens':
            epoch_predictions = np.zeros((len(X_train), num_classes))
            epoch_execmask = np.zeros(len(X_train)) # Which inputs were executed.
            training_targets = floatX(training_targets)

        # Training pass.

        start_time = time.time()
        train_err, train_n = 0., 0.

        learning_rate = rampup_value * rampdown_value * config.learning_rate_max
        adam_beta1 = rampdown_value * config.adam_beta1 + (1.0 - rampdown_value) * config.rampdown_beta1_target
        unsup_weight = rampup_value * scaled_unsup_weight_max
        if epoch == config.start_epoch:
            unsup_weight = 0.0

        with thread_utils.ThreadPool(8) as thread_pool:
            if config.network_type == 'pi':
                minibatches = iterate_minibatches_augment_pi(X_train, y_train, mask_train, config.minibatch_size)
                minibatches = thread_utils.run_iterator_concurrently(minibatches, thread_pool)
                for (n, indices, inputs_a, inputs_b, labels, mask) in minibatches: 
                    (e_train, ) = train_fn(inputs_a, inputs_b, labels, mask, floatX(learning_rate), floatX(adam_beta1), floatX(unsup_weight))
                    train_err += e_train * n
                    train_n += n
            elif config.network_type == 'tempens':
                minibatches = iterate_minibatches_augment_tempens(X_train, y_train, mask_train, training_targets, config.minibatch_size)
                minibatches = thread_utils.run_iterator_concurrently(minibatches, thread_pool)
                for (n, indices, inputs, labels, mask, targets) in minibatches: 
                    (e_train, prediction) = train_fn(inputs, labels, mask, targets, floatX(learning_rate), floatX(adam_beta1), floatX(unsup_weight))
                    for i, j in enumerate(indices):
                        epoch_predictions[j] = prediction[i] # Gather epoch predictions.
                        epoch_execmask[j] = 1.0
                    train_err += e_train * n
                    train_n += n

        # Test pass.

        val_err, val_acc, val_n = 0., 0., 0.
        with thread_utils.ThreadPool(8) as thread_pool:
            minibatches = iterate_minibatches(X_test, y_test, config.minibatch_size)
            minibatches = thread_utils.run_iterator_concurrently(minibatches, thread_pool)
            for (n, inputs, labels) in minibatches: 
                err, acc = val_fn(inputs, labels)
                val_err += err * n
                val_acc += acc * n
                val_n += n

        if config.network_type == 'tempens':
            if config.max_unlabeled_per_epoch is None:
                # Basic mode.
                ensemble_prediction = (config.prediction_decay * ensemble_prediction) + (1.0 - config.prediction_decay) * epoch_predictions
                training_targets = ensemble_prediction / (1.0 - config.prediction_decay ** ((epoch - config.start_epoch) + 1.0))
            else:
                # Sparse updates.
                epoch_execmask = epoch_execmask.reshape(-1, 1)
                ensemble_prediction = epoch_execmask * (config.prediction_decay * ensemble_prediction + (1.0 - config.prediction_decay) * epoch_predictions) + (1.0 - epoch_execmask) * ensemble_prediction
                training_targets = ensemble_prediction / (np.sum(ensemble_prediction, axis=1, keepdims=True) + 1e-8) # Normalize

        # Export stats.

        training_csv.add_data(
            epoch,
            time.time() - start_time,
            train_err / train_n,
            val_err / val_n,
            val_acc / val_n * 100.0,
            learning_rate)

        # Export progress monitor data.

        if monitor_filename is not None:
            with open(monitor_filename, 'wt') as f:
                json.dump({"loss": 1.0 - val_acc / val_n, "cur_epoch": (epoch + 1), "max_epoch": config.num_epochs}, f)

        # Print stats.

        print("Epoch %3d of %3d took %6.3fs   Loss %.7f, %.7f  Acc=%5.2f  LR=%.7f" % (
            epoch, 
            config.num_epochs, 
            time.time() - start_time, 
            train_err / train_n,
            val_err / val_n, 
            val_acc / val_n * 100.0,
            learning_rate))

    #----------------------------------------------------------------------------
    # Save and exit.
    #----------------------------------------------------------------------------

    training_csv.close()
    print("Saving the final network.")
    np.savez(os.path.join(result_subdir, 'network-final.npz'), *lasagne.layers.get_all_param_values(net))
    save_network(net, os.path.join(result_subdir, 'network-final.pkl'))
    print("Done.")

###################################################################################################
# Bootstrap.
###################################################################################################

if __name__ == "__main__":
    print "Starting up..."
    run_training()
    print "Exiting..."

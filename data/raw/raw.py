# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import subprocess
import pickle
import torch
import os
import argparse
import gzip

parser = argparse.ArgumentParser(description='Continuum learning')

# experiment parameters
parser.add_argument('dset', choices=['all', 'mnist', 'fmnist', 'cifar10', 'cifar100'], type=str, default='all',
                    help='Which dataset to download.')
args = parser.parse_args()

cifar_path = "cifar-100-python.tar.gz"
cifar10_path = "cifar-10-python.tar.gz"
mnist_path = "mnist.npz"

fmnist_train_samples_path = "train-images-idx3-ubyte.gz"
fmnist_train_labels_path = "train-labels-idx1-ubyte.gz"
fmnist_test_samples_path = "t10k-images-idx3-ubyte.gz"
fmnist_test_labels_path = "t10k-labels-idx1-ubyte.gz"


def load_fmnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                                '%s-labels-idx1-ubyte.gz'
                                % kind)
    images_path = os.path.join(path,
                                '%s-images-idx3-ubyte.gz'
                                % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                                offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                offset=16).reshape(len(labels), 784)

    return images, labels


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if args.dset == 'all' or args.dset == 'cifar100':
    # URL from: https://www.cs.toronto.edu/~kriz/cifar.html
    if not os.path.exists(cifar_path):
        subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz", shell=True)

    subprocess.call("tar xzfv cifar-100-python.tar.gz", shell=True)

    cifar100_train = unpickle('cifar-100-python/train')
    cifar100_test = unpickle('cifar-100-python/test')

    x_tr = torch.from_numpy(cifar100_train[b'data'])
    y_tr = torch.LongTensor(cifar100_train[b'fine_labels'])
    x_te = torch.from_numpy(cifar100_test[b'data'])
    y_te = torch.LongTensor(cifar100_test[b'fine_labels'])

    torch.save((x_tr, y_tr, x_te, y_te), 'cifar100.pt')

if args.dset == 'all' or args.dset == 'cifar10':
    # URL from: https://www.cs.toronto.edu/~kriz/cifar.html
    if not os.path.exists(cifar10_path):
        subprocess.call("wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", shell=True)

    subprocess.call("tar xzfv cifar-10-python.tar.gz", shell=True)

    x_tr = None
    for batch in range(5):  # only two batches
        cifar10_train = unpickle('cifar-10-batches-py/data_batch_' + str(batch + 1))
        if x_tr is None:
            x_tr = torch.from_numpy(cifar10_train[b'data'])
            y_tr = torch.LongTensor(cifar10_train[b'labels'])
        else:
            x_tr = torch.cat((x_tr, torch.from_numpy(cifar10_train[b'data'])), 0)
            y_tr = torch.cat((y_tr, torch.LongTensor(cifar10_train[b'labels'])), 0)

    cifar10_test = unpickle('cifar-10-batches-py/test_batch')
    print("cifar 10 train size is ", y_tr.size(0))

    x_te = torch.from_numpy(cifar10_test[b'data'])

    y_te = torch.LongTensor(cifar10_test[b'labels'])
    x_te = x_te[0:1000]
    y_te = y_te[0:1000]
    torch.save((x_tr, y_tr, x_te, y_te), 'cifar10.pt')

if args.dset == 'all' or args.dset == 'mnist':
    # URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
    if not os.path.exists(mnist_path):
        subprocess.call("! wget https://s3.amazonaws.com/img-datasets/mnist.npz", shell=True)

    f = np.load('mnist.npz')
    x_tr = torch.from_numpy(f['x_train'])
    y_tr = torch.from_numpy(f['y_train']).long()
    x_te = torch.from_numpy(f['x_test'])
    y_te = torch.from_numpy(f['y_test']).long()
    f.close()

    torch.save((x_tr, y_tr), 'mnist_train.pt')
    torch.save((x_te, y_te), 'mnist_test.pt')

if args.dset == 'all' or args.dset == 'fmnist':
    # URL from: https://github.com/fchollet/keras/blob/master/keras/datasets/mnist.py
    if not os.path.exists(fmnist_train_samples_path):
        subprocess.call("! wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", shell=True)
    if not os.path.exists(fmnist_train_labels_path):
        subprocess.call("! wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", shell=True)
    if not os.path.exists(fmnist_test_samples_path):
        subprocess.call("! wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", shell=True)
    if not os.path.exists(fmnist_test_labels_path):
        subprocess.call("! wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", shell=True)

    X_train, y_train = load_fmnist('./', kind='train') #(60000, 784), (60000,)
    X_test, y_test = load_fmnist('./', kind='t10k')    #(10000, 784), (10000,)

    x_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).long()
    x_te = torch.from_numpy(X_test)
    y_te = torch.from_numpy(y_test).long()
    f.close()

    torch.save((x_tr, y_tr), 'mnist_train.pt')
    torch.save((x_te, y_te), 'mnist_test.pt')



print("Finished downloads raw data.")

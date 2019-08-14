#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from singa import singa_wrap as singa
from singa import autograd
from singa import tensor
from singa import device
from singa import opt
import numpy as np
from tqdm import trange
import os
import urllib.request
import gzip
import codecs
try:
    import pickle
except ImportError:
    import cPickle as pickle
import time

class CNN:
    def __init__(self):
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = autograd.flatten(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = self.linear2(y)
        return y

def load_dataset():
    train_x_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_y_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    valid_x_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    valid_y_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    train_x = read_image_file(check_exist_or_download(train_x_url)).astype(
        np.float32)
    train_y = read_label_file(check_exist_or_download(train_y_url)).astype(
        np.float32)
    valid_x = read_image_file(check_exist_or_download(valid_x_url)).astype(
        np.float32)
    valid_y = read_label_file(check_exist_or_download(valid_y_url)).astype(
        np.float32)
    return train_x, train_y, valid_x, valid_y


def check_exist_or_download(url):

    download_dir = '/tmp/'

    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename


def read_label_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape(
            (length))
        return parsed


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
            (length, 1, num_rows, num_cols))
        return parsed

def to_categorical(y, num_classes):
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical


# Function to all reduce Accuracy and Loss from Multiple Devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    singa.synch(reducer.data, dist_opt.communicator)
    output=tensor.to_numpy(reducer)
    return output

def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum()

def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num,:,:,:] = xpad[data_num, :, offset[0]: offset[0] + 28, offset[1]: offset[1] + 28]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x

def data_partition(dataset_x, dataset_y, rank_in_global, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = rank_in_global * data_per_rank
    idx_end = (rank_in_global + 1) * data_per_rank
    return dataset_x[idx_start: idx_end], dataset_y[idx_start: idx_end]

def sychronize(tensor, dist_opt):
    singa.synch(tensor.data, dist_opt.communicator)
    tensor /= dist_opt.world_size


# create model
model = CNN()

sgd = opt.SGD(lr=0.04, momentum=0.9, weight_decay=1e-5)
sgd = opt.DistOpt(sgd)
dev = device.create_cuda_gpu_on(sgd.rank_in_local)


# Prepare training and valadiation data
train_x, train_y, test_x, test_y = load_dataset()
IMG_SIZE = 28
num_classes=10    
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)    

# Normalization
train_x = train_x / 255
test_x = test_x / 255


train_x, train_y = data_partition(train_x, train_y, sgd.rank_in_global, sgd.world_size)
test_x, test_y = data_partition(test_x, test_y, sgd.rank_in_global, sgd.world_size)

max_epoch = 10
batch_size = 64
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
num_train_batch = train_x.shape[0] // batch_size
num_test_batch = test_x.shape[0] // batch_size
idx = np.arange(train_x.shape[0], dtype=np.int32)


#Sychronize the initial parameter
autograd.training = True
x = np.random.randn(batch_size, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
y = np.zeros( shape=(batch_size, num_classes), dtype=np.int32)
tx.copy_from_numpy(x)
ty.copy_from_numpy(y)
out = model.forward(tx)
loss = autograd.softmax_cross_entropy(out, ty)               
for p, g in autograd.backward(loss):
    sychronize(p, sgd)


# Training and Evaulation Loop
for epoch in range(max_epoch):
    start_time = time.time()
    np.random.shuffle(idx)

    if(sgd.rank_in_global==0):
        print('Starting Epoch %d:' % (epoch))
    
    # Training Phase
    autograd.training = True
    train_correct = np.zeros(shape=[1],dtype=np.float32)
    test_correct = np.zeros(shape=[1],dtype=np.float32)
    train_loss = np.zeros(shape=[1],dtype=np.float32)
    
    for b in range(num_train_batch):
        x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
        x = augmentation(x, batch_size)
        y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        out = model.forward(tx)
        loss = autograd.softmax_cross_entropy(out, ty)               
        train_correct += accuracy(tensor.to_numpy(out), y)
        train_loss += tensor.to_numpy(loss)[0]
        for p, g in autograd.backward(loss):
            sgd.update(p, g)

    # Reduce the Evaluation Accuracy and Loss from Multiple Devices
    reducer = tensor.Tensor((1,), dev, tensor.float32)
    train_correct = reduce_variable(train_correct, sgd, reducer)
    train_loss = reduce_variable(train_loss, sgd, reducer)

    # Output the Training Loss and Accuracy
    if(sgd.rank_in_global==0):
        print('Training loss = %f, training accuracy = %f' % (train_loss, train_correct / (num_train_batch*batch_size*sgd.world_size)), flush=True)

    # Evaluation Phase
    autograd.training = False
    for b in range(num_test_batch):
        x = test_x[b * batch_size: (b + 1) * batch_size]
        y = test_y[b * batch_size: (b + 1) * batch_size]
        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)
        out_test = model.forward(tx)
        test_correct += accuracy(tensor.to_numpy(out_test), y)

    # Reduce the Evaulation Accuracy from Multiple Devices
    test_correct = reduce_variable(test_correct, sgd, reducer)

    # Output the Evaluation Accuracy
    if(sgd.rank_in_global==0):
        print('Evaluation accuracy = %f, Elapsed Time = %fs' % 
              (test_correct / (num_test_batch*batch_size*sgd.world_size), time.time() - start_time ), flush=True)

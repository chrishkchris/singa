
# SINGA Distributed Module: Demostration of Convolutional Neural Network

## Introduction to the Distributed Module:

### 1. SINGA can utilize the Message Passing Interface (MPI) to start up and coordinate Multiple Processes to parallelize the training process:

![MPI.png](attachment:MPI.png)

### The MPI starts up multi-processes for the distributed training. Each process runs a training script which utilizes one GPU for training. Each process has an individual rank, which gives information of which GPU the individual process is using. The training data is partitioned, so that each process can evaluate the sub-gradient based on the partitioned training data. Once the sub-graident is calculated on each processes, the overall stochastic gradient is obtained by all-reduce the sub-gradients evaluated by all processes. The all-reduce operation is supported by the NVidia Collective Communication Library (NCCL).   

### 2. The all-reduce operation by NCCL can be used to reduce and synchronize the parameters from different GPUs. Let's consider a data partitioned distributed training using 4 GPUs. Once the sub-gradients from the 4 GPUs are calculated, the NCCL can perform the all-reduce process so that all the GPUs can get the sum of the sub-gradients over the GPUs:

 ![AllReduce.png](attachment:AllReduce.png)

### Finally, the parameter update of Stochastic Gradient Descent (SGD) can then be performed by using the overall stochastic gradient obtained by the all-reduce process.

## Procedures for Distributed Training:

### 1. Import SINGA and Miscellaneous Libraries used for the Tutorial


```python
from singa import singa_wrap as singa
from singa import autograd
from singa import tensor
from singa import device
from singa import opt
import numpy as np
import os
import sys
import gzip
import codecs
import time
import urllib.request
```

### 2. Create a Convolutional Neural Network Model


```python
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

# create model
model = CNN()
```

### 3. Create a Distributed Optimizer Object and Device Assignment


```python
sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
sgd = opt.DistOpt(sgd)
dev = device.create_cuda_gpu_on(sgd.rank_in_local)
```

### 4. Prepare the Training and Evaluation Data


```python
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

    
# Prepare training and valadiation data
train_x, train_y, test_x, test_y = load_dataset()
IMG_SIZE = 28
num_classes=10    
train_y = to_categorical(train_y, num_classes)
test_y = to_categorical(test_y, num_classes)    

# Normalization
train_x = train_x / 255
test_x = test_x / 255

print((train_x.shape))
print((train_y.shape))
print((test_x.shape))
print((test_y.shape))
```

    (60000, 1, 28, 28)
    (60000, 10)
    (10000, 1, 28, 28)
    (10000, 10)
    

### 5. Data Partitioning of the Training and Evaluation Datasets


```python
def data_partition(dataset_x, dataset_y, rank_in_global, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = rank_in_global * data_per_rank
    idx_end = (rank_in_global + 1) * data_per_rank
    return dataset_x[idx_start: idx_end], dataset_y[idx_start: idx_end]

train_x, train_y = data_partition(train_x, train_y, sgd.rank_in_global, sgd.world_size)
test_x, test_y = data_partition(test_x, test_y, sgd.rank_in_global, sgd.world_size)
```

### 6. Configuring the Training Loop Variables


```python
max_epoch = 10
batch_size = 64
tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
num_train_batch = train_x.shape[0] // batch_size
num_test_batch = test_x.shape[0] // batch_size
idx = np.arange(train_x.shape[0], dtype=np.int32)
```

### 7. Initialize and Synchronize the Model Parameters


```python
def sychronize(tensor, dist_opt):
    dist_opt.all_reduce(tensor.data)
    tensor /= dist_opt.world_size

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
```

### 8. Start the Training and Evaluation Loop


```python
# Function to all reduce Accuracy and Loss from Multiple Devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
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
        sgd.backward_and_update(loss)

    # Reduce the Evaluation Accuracy and Loss from Multiple Devices
    reducer = tensor.Tensor((1,), dev, tensor.float32)
    train_correct = reduce_variable(train_correct, sgd, reducer)
    train_loss = reduce_variable(train_loss, sgd, reducer)

    # Output the Training Loss and Accuracy
    if(sgd.rank_in_global==0):
        print('Training loss = %f, training accuracy = %f' % 
              (train_loss, train_correct / (num_train_batch*batch_size*sgd.world_size)), flush=True)

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
```

    Starting Epoch 0:
    Training loss = 586.781189, training accuracy = 0.793390
    Evaluation accuracy = 0.928085, Elapsed Time = 2.801320s
    Starting Epoch 1:
    Training loss = 231.525131, training accuracy = 0.923026
    Evaluation accuracy = 0.958233, Elapsed Time = 2.765840s
    Starting Epoch 2:
    Training loss = 167.739914, training accuracy = 0.943970
    Evaluation accuracy = 0.971554, Elapsed Time = 2.783665s
    Starting Epoch 3:
    Training loss = 134.832535, training accuracy = 0.954809
    Evaluation accuracy = 0.973057, Elapsed Time = 2.790509s
    Starting Epoch 4:
    Training loss = 117.657333, training accuracy = 0.960679
    Evaluation accuracy = 0.969852, Elapsed Time = 2.805864s
    Starting Epoch 5:
    Training loss = 106.407524, training accuracy = 0.964298
    Evaluation accuracy = 0.974259, Elapsed Time = 2.799248s
    Starting Epoch 6:
    Training loss = 95.293930, training accuracy = 0.968083
    Evaluation accuracy = 0.979968, Elapsed Time = 2.797730s
    Starting Epoch 7:
    Training loss = 89.077400, training accuracy = 0.970051
    Evaluation accuracy = 0.971755, Elapsed Time = 2.795567s
    Starting Epoch 8:
    Training loss = 80.842201, training accuracy = 0.972685
    Evaluation accuracy = 0.979567, Elapsed Time = 2.791101s
    Starting Epoch 9:
    Training loss = 78.718658, training accuracy = 0.973269
    Evaluation accuracy = 0.982672, Elapsed Time = 2.800081s
    

## Control the Number of GPUs for Training:
### The Training above used only 1 Process GPU for the Training Process.
### To Control the Number of GPUs used for the Training, Change the hostfile for MPI.
### The hostfile below uses 4 Processes and hence 4 GPUs for the Training.


```python
!cat host_file
```

    localhost:4

### Now, use the MPIEXEC command to Execute the Multi-GPUs Training with the hostfile:


```python
!/home/ubuntu/mpich-3.3/build/bin/mpiexec --hostfile host_file python3 mnist_dist_demo.py
```

    Starting Epoch 0:
    Training loss = 673.246277, training accuracy = 0.760517
    Evaluation accuracy = 0.930489, Elapsed Time = 0.757460s
    Starting Epoch 1:
    Training loss = 240.009323, training accuracy = 0.919705
    Evaluation accuracy = 0.964042, Elapsed Time = 0.707835s
    Starting Epoch 2:
    Training loss = 168.806030, training accuracy = 0.944010
    Evaluation accuracy = 0.967448, Elapsed Time = 0.710606s
    Starting Epoch 3:
    Training loss = 139.131454, training accuracy = 0.953676
    Evaluation accuracy = 0.971755, Elapsed Time = 0.710840s
    Starting Epoch 4:
    Training loss = 117.479889, training accuracy = 0.960487
    Evaluation accuracy = 0.974659, Elapsed Time = 0.711388s
    Starting Epoch 5:
    Training loss = 103.085609, training accuracy = 0.965812
    Evaluation accuracy = 0.979267, Elapsed Time = 0.712624s
    Starting Epoch 6:
    Training loss = 97.565521, training accuracy = 0.966897
    Evaluation accuracy = 0.979868, Elapsed Time = 0.714128s
    Starting Epoch 7:
    Training loss = 86.971985, training accuracy = 0.970903
    Evaluation accuracy = 0.979868, Elapsed Time = 0.715277s
    Starting Epoch 8:
    Training loss = 79.487328, training accuracy = 0.973341
    Evaluation accuracy = 0.982372, Elapsed Time = 0.715577s
    Starting Epoch 9:
    Training loss = 74.658951, training accuracy = 0.974793
    Evaluation accuracy = 0.982672, Elapsed Time = 0.717571s
    

### It results in several times speed up compared to the single GPU training. 

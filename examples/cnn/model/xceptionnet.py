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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

# the code is modified from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py

from singa import autograd
from singa import layer
from singa import model


class Block(layer.Layer):

    def __init__(self,
                 in_filters,
                 out_filters,
                 reps,
                 strides=1,
                 padding=0,
                 start_with_relu=True,
                 grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = layer.Conv2d(in_filters,
                                     out_filters,
                                     1,
                                     stride=strides,
                                     padding=padding,
                                     bias=False)
            self.skipbn = layer.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.layers = []

        filters = in_filters
        if grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(
                layer.SeparableConv2d(in_filters,
                                      out_filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            self.layers.append(autograd.ReLU())
            self.layers.append(
                layer.SeparableConv2d(filters,
                                      filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(filters))

        if not grow_first:
            self.layers.append(autograd.ReLU())
            self.layers.append(
                layer.SeparableConv2d(in_filters,
                                      out_filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(out_filters))

        if not start_with_relu:
            self.layers = self.layers[1:]
        else:
            self.layers[0] = autograd.ReLU()

        if strides != 1:
            self.layers.append(layer.MaxPool2d(3, strides, padding + 1))       

        def register_layers(self, *args):
            if len(args) == 1 and isinstance(args[0], OrderedDict):
                for key, value in args[0].items():
                    self._layers[name] = value
            else:
                for idx, value in enumerate(args):
                    value.name = str(idx)
                    self._layers[value.name] = value
                    value.__dict__['_parent'] = self

        register_layers(self.layers)

    def __call__(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            if isinstance(y, tuple):
                y = y[0]
            y = layer(y)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        y = autograd.add(y, skip)
        return y
    
'''
    def set_params(self, parameters):
        super().set_params(parameters)
        for lyr in self.layers:
            lyr.set_params(parameters)
    
    def get_params(self):
        params = super().get_params()
        for lyr in self.layers:
            params.update(lyr.get_params())
        return params
'''

class Xception(model.Model):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=10, num_channels=3):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.input_size = 299
        self.dimension = 4

        self.conv1 = layer.Conv2d(num_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = layer.BatchNorm2d(32)

        self.conv2 = layer.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = layer.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64,
                            128,
                            2,
                            2,
                            padding=0,
                            start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128,
                            256,
                            2,
                            2,
                            padding=0,
                            start_with_relu=True,
                            grow_first=True)
        self.block3 = Block(256,
                            728,
                            2,
                            2,
                            padding=0,
                            start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)

        self.block8 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)

        self.block12 = Block(728,
                             1024,
                             2,
                             2,
                             start_with_relu=True,
                             grow_first=False)

        self.conv3 = layer.SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = layer.BatchNorm2d(1536)

        # do relu here
        self.conv4 = layer.SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = layer.BatchNorm2d(2048)

        self.globalpooling = layer.MaxPool2d(10, 1)
        self.fc = layer.Linear(2048, num_classes)

        #register_layers(self.layers)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = autograd.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = autograd.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = autograd.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = autograd.relu(features)
        x = self.globalpooling(x)
        x = autograd.flatten(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a Xceptionnet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = Xception(**kwargs)

    return model


__all__ = ['Xception', 'create_model']

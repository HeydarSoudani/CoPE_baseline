# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
import torchvision.models as models

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):

            if i < (len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))

                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        print("BIAS IS", bias)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20, bias=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, bias)


class Conv_4(nn.Module):
    def __init__(self, args):
        super(Conv_4, self).__init__()
		
        if args.dataset in ['mnist', 'fmnist']:
            img_channels = 1	  	# 1
            self.last_layer = 1 	# 3 for 3-layers - 1 for 4-layers
            self.tensor_shape = (1, 28, 28)
        elif args.dataset in ['cifar10', 'cifar100']:
            img_channels = 3	  	# 3 
            self.last_layer = 2 	# 4 for 3-layers - 2 for 4-layers
            self.tensor_shape = (3, 32, 32)

        self.filters_length = 256    # 128 for 3-layers - 256 for 4-layers

        self.layer1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            # nn.ReLU(),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2), #input: 28 * 28 * 3, output: 28 * 28 * 32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 28 * 28 * 32, output: 14 * 14 * 32
            nn.Dropout(args.dropout)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), #input: 14 * 14 * 32, output: 14 * 14 * 64
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=2), #input: 14 * 14 * 64, output: 14 * 14 * 64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 14 * 14 * 64, output: 7* 7 * 64
            nn.Dropout(args.dropout)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #input: 7 * 7 * 64, output: 7 * 7 * 128
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #input: 7 * 7 * 128, output: 7 * 7 * 128
            nn.BatchNorm2d(128),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 7 * 7 * 128, output: 3* 3 * 128
            nn.Dropout(args.dropout)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #input: 3 * 3 * 128, output: 3 * 3 * 256
            nn.PReLU(),
            # nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #input: 3*3*256, output: 3*3*256
            nn.BatchNorm2d(256),
            nn.PReLU(),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),   #input: 3*3*256, output: 1*1*256
            nn.Dropout(args.dropout)
        )

        self.ip1 = nn.Linear(self.filters_length*self.last_layer*self.last_layer, args.hidden_dims)
        self.preluip1 = nn.PReLU()
        self.dropoutip1 = nn.Dropout(args.dropout)
        self.classifier = nn.Linear(args.hidden_dims, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], *self.tensor_shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(-1, self.filters_length*self.last_layer*self.last_layer)

        features = self.preluip1(self.ip1(x))
        x = self.dropoutip1(features)
        logits = self.classifier(x)
        
        # return logits, features
        return logits

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.device = args[0] # store device

        self.layer1 = self.layer1.to(*args, **kwargs)
        self.layer2 = self.layer2.to(*args, **kwargs)
        self.layer3 = self.layer3.to(*args, **kwargs)
        self.layer4 = self.layer4.to(*args, **kwargs)

        self.ip1 = self.ip1.to(*args, **kwargs)
        self.preluip1 = self.preluip1.to(*args, **kwargs)
        self.dropoutip1 = self.dropoutip1.to(*args, **kwargs)
        self.classifier = self.classifier.to(*args, **kwargs)
        return self

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)


class Resnet50(nn.Module):
    def __init__(self, args):
        super(Resnet50, self).__init__()

        self.pretrained = models.resnet50(pretrained=True)
        self.fc1 = nn.Linear(1000, args.n_hiddens)
        self.dp1 = nn.Dropout(args.dropout)
        self.fc2 = nn.Linear(args.n_hiddens, args.n_outputs)
        self.dp2 = nn.Dropout(args.dropout)

        # init the fc layers
        self.pretrained.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.pretrained.fc.bias.data.zero_()
        self.fc1.apply(Xavier)
        self.fc2.apply(Xavier)

    def forward(self, x):
        # x = x.view(x.size(0), -1)
        x = x.view(x.size(0), 3, 32, 32)
        print(x.shape)
        x = self.pretrained(x)
        x = self.dp1(torch.relu(x))
        features = torch.relu(self.fc1(x))
        out = self.fc2(self.dp2(features))
        # return out, features
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)





import torch
from torch import nn as nn
import torch.nn.functional as F

from escnn import gspaces
from escnn import nn as enn

from typing import Tuple

def conv7x7(in_type: enn.FieldType, out_type: enn.FieldType, stride=2, padding=3,
            dilation=1, bias=False):
    """7x7 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 7,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

def conv1x1(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=0,
            dilation=1, bias=False):
    """1x1 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 1,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

class EqBasicBlock(enn.EquivariantModule):
    # expansion = 1

    def __init__(self, in_type, out_type, stride=1, downsample=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.conv1 = conv3x3(in_type, out_type, stride=stride, padding=1)
        self.bn1 = enn.InnerBatchNorm(self.conv1.out_type)
        self.relu = enn.ReLU(self.conv1.out_type)
        self.conv2 = conv3x3(self.conv1.out_type, out_type, stride=1, padding=1)
        self.bn2 = enn.InnerBatchNorm(self.conv2.out_type)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        if self.shortcut is not None:
            return self.shortcut.evaluate_output_shape(input_shape)
        else:
            return input_shape
    
    # def evaluate_output_shape(self, input_shape):
    #     """
    #     input_shape: (batch, channels, height, width)
    #     returns the output shape of this module
    #     """
    #     # Compute shape after conv1
    #     shape = self.conv1.evaluate_output_shape(input_shape)
    #     # conv2
    #     shape = self.conv2.evaluate_output_shape(shape)
    #     # If downsample exists, its output shape must match
    #     if self.downsample is not None:
    #         shape = self.downsample.evaluate_output_shape(input_shape)
    #     return shape

class EqResNet18(nn.Module):
    def __init__(self, N=4, projector_hidden_size=1024, n_classes=128):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = gspaces.rot2dOnR2(N)

        # if N != 1:
        #     self.r2_act = gspaces.flipRot2dOnR2(N)
        # else:
        #     self.r2_act = gspaces.flip2dOnR2()

        # input type: 3-channel RGB image
        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat64 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 64)
        self.feat128 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 128)
        self.feat256 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 256)
        self.feat512 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * 512)

        # initial conv + BN + ReLU
        #self.conv1 = conv7x7(self.in_type, self.feat64, kernel_size=7, stride=2, padding=3)
        self.conv1 = enn.R2Conv(self.in_type, self.feat64, kernel_size=7, stride=2, padding=3)
        self.bn1 = enn.InnerBatchNorm(self.feat64)
        self.relu = enn.ReLU(self.feat64)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = enn.PointwiseMaxPool2D(self.feat64, kernel_size=3, stride=2, padding=1)

        # ResNet layers
        # self.layer1 = self._make_layer(self.feat64, self.feat64, blocks=2)
        # self.layer2 = self._make_layer(self.feat64, self.feat128, blocks=2, stride=2)
        # self.layer3 = self._make_layer(self.feat128, self.feat256, blocks=2, stride=2)
        # self.layer4 = self._make_layer(self.feat256, self.feat512, blocks=2, stride=2)
        self.layer1 = self._make_layer(self.maxpool.out_type, self.feat64, blocks=2)
        self.layer2 = self._make_layer(self.layer1.out_type, self.feat128, blocks=2, stride=2)
        self.layer3 = self._make_layer(self.layer2.out_type, self.feat256, blocks=2, stride=2)
        self.layer4 = self._make_layer(self.layer3.out_type, self.feat512, blocks=2, stride=2)
        
        # Pooling
        #self.pool = enn.PointwiseMaxPool2D(self.layer4.out_type, kernel_size=3, stride=1, padding=0)
        self.gpool = enn.GroupPooling(self.layer4.out_type)

        # Fully connected
        c = self.gpool.out_type.size
        
        self.fully_net = nn.Sequential(
            nn.Linear(c, projector_hidden_size),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_size, n_classes),
        )

    def _make_layer(self, in_type, out_type, blocks, stride=1):
        layers = []
        downsample = None

        if stride != 1 or in_type != out_type:
            downsample = conv1x1(in_type, out_type, stride=stride, bias=False)

        layers.append(EqBasicBlock(in_type, out_type, stride, downsample))
        for _ in range(1, blocks):
            layers.append(EqBasicBlock(out_type, out_type))
        
        return enn.SequentialModule(*layers)
    
    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = enn.GeometricTensor(self.maxpool(x.tensor), x.type)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.pool(x)
        x = self.gpool(x)

        x = x.tensor
        b, c, w, h = x.shape
        hidden = F.avg_pool2d(x, (w, h))
        hidden = hidden.view(hidden.size(0), -1)
        hidden = torch.flatten(hidden, 1)
        
        z = self.fully_net(hidden)

        return hidden, z
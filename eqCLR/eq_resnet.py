import torch
from torch import nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, Bottleneck
import math
import numpy as np

from escnn import gspaces
from escnn import nn as enn
from escnn.group import SO2
from torchvision.models import resnet18

from typing import Tuple

def convkxk(k: int,in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=False):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, k,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )

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

def conv4x4(in_type: enn.FieldType, out_type: enn.FieldType, stride=2, padding=1,
            dilation=1, bias=False):
    """4x4 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 4,
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

    def __init__(self, in_type, out_type, stride=1, downsample=None, eq_downsampling=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.eq_downsampling = eq_downsampling
        
        if stride != 1 and eq_downsampling == "kernel_size":
            self.conv1 = conv4x4(in_type, out_type, stride=stride, padding=1)
        else:
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
        pass
        # assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        # if self.shortcut is not None:
        #     return self.shortcut.evaluate_output_shape(input_shape)
        # else:
        #     return input_shape

class EqBasicBlock_hue(enn.EquivariantModule):
    # expansion = 1

    def __init__(self, in_type, out_type, stride=1, downsample=None, eq_downsampling=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.eq_downsampling = eq_downsampling
        
        if stride != 1 and eq_downsampling == "kernel_size":
            self.conv1 = enn.R2HueGroupConv(in_type, out_type, 4,
                      stride=stride,
                      padding=1,
                      dilation=1,
                      bias=False,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )
        else:
            self.conv1 = enn.R2HueGroupConv(in_type, out_type, 3,
                      stride=stride,
                      padding=1,
                      dilation=1,
                      bias=False,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )
        self.bn1 = enn.InnerBatchNorm(self.conv1.out_type)
        self.relu = enn.ReLU(self.conv1.out_type)
        self.conv2 = enn.R2HueGroupConv(self.conv1.out_type, out_type, 3,
                      stride=1,
                      padding=1,
                      dilation=1,
                      bias=False,
                      sigma=None,
                      frequencies_cutoff=lambda r: 3*r,
                      )
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
        pass
        # assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        # if self.shortcut is not None:
        #     return self.shortcut.evaluate_output_shape(input_shape)
        # else:
        #     return input_shape

class EqBasicBlock_SO2(enn.EquivariantModule):
    # expansion = 1

    def __init__(self, in_type, out_type, stride=1, downsample=None, eq_downsampling=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.eq_downsampling = eq_downsampling
        
        if stride != 1 and eq_downsampling == "kernel_size":
            self.conv1 = conv4x4(in_type, out_type, stride=stride, padding=1)
        else:
            self.conv1 = conv3x3(in_type, out_type, stride=stride, padding=1)
        self.bn1 = enn.IIDBatchNorm2d(self.conv1.out_type)
        self.relu = enn.FourierELU(self.conv1.out_type)
        self.conv2 = conv3x3(self.conv1.out_type, out_type, stride=1, padding=1)
        self.bn2 = enn.IIDBatchNorm2d(self.conv2.out_type)
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
    
class EqBasicBlock_v2(enn.EquivariantModule):
    # expansion = 1

    def __init__(self, in_type, n_channels, irreps, stride=1, downsample=None, N_grid=16):
        super().__init__()
        self.in_type = in_type
        self.n_channels = n_channels
        self.gspace = in_type.gspace
        self.irreps = irreps
        self.N_grid = N_grid
        
        self.relu = enn.FourierELU(self.gspace, self.n_channels, irreps=self.irreps, N=self.N_grid)
        in_type_relu = self.relu.in_type
        self.out_type = self.relu.out_type

        self.conv1 = conv3x3(in_type, in_type_relu, stride=stride, padding=1)
        self.bn1 = enn.IIDBatchNorm2d(in_type_relu)
        # self.relu = enn.FourierELU(self.gspace, self.n_channels, irreps=self.irreps)
        self.conv2 = conv3x3(self.out_type, self.out_type, stride=1, padding=1)
        self.bn2 = enn.IIDBatchNorm2d(self.out_type)
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
        pass
        # assert len(input_shape) == 4
        # assert input_shape[1] == self.in_type.size
        # if self.shortcut is not None:
        #     return self.shortcut.evaluate_output_shape(input_shape)
        # else:
        #

class EqBootleneck(enn.EquivariantModule):
    def __init__(self, in_type, out_type, stride=1, downsample=None, eq_downsampling=None):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        self.eq_downsampling = eq_downsampling
        
        self.conv1 = conv1x1(in_type, out_type)
        self.bn1 = enn.InnerBatchNorm(self.conv1.out_type)
        self.relu1 = enn.ReLU(self.conv1.out_type)
        self.conv2 = conv3x3(self.conv1.out_type, out_type, stride=stride, padding=1)
        self.bn2 = enn.InnerBatchNorm(self.conv2.out_type)
        self.relu2 = enn.ReLU(self.conv2.out_type)
        self.conv3 = conv1x1(self.conv2.out_type, out_type)
        self.bn3 = enn.InnerBatchNorm(self.conv3.out_type)
        self.relu3 = enn.ReLU(self.conv3.out_type)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu3(out)
        return out
    
    def evaluate_output_shape(self, input_shape: Tuple):
        pass


def eq_resnet(depth, **kwargs):
    if depth == 18:
        layers = [2,2,2,2]
    elif depth == 33:
        layers = [3,3,4,3]
    elif depth == 34:
        layers = [3,4,6,3]
    else:
        raise ValueError(f"Unsupported depth: {depth}")
    return EqResNet(layers=layers, **kwargs)

class EqResNet(nn.Module):
    def __init__(self, N=4, layers=[2, 2, 2, 2], block='basic', eq_blocks=4, projector_hidden_size=1024, n_classes=128, maxpool=True, adjust_channels='keep_param'):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = gspaces.rot2dOnR2(N)

        if block == 'basic':
            eq_block = EqBasicBlock
            torch_block = BasicBlock
        elif block == 'bottleneck':
            eq_block = EqBootleneck
            torch_block = Bottleneck
        else:
            raise ValueError(f"Unsupported block type: {block}. Only 'basic' and 'bottleneck' are supported.")

        assert 0 <= eq_blocks <= 4
        self.eq_blocks = eq_blocks
        self.maxpool = maxpool

        # Normalization of number of independent channels
        if adjust_channels == 'keep_channels':
            self.S = self.r2_act.fibergroup.order()
        elif adjust_channels == 'keep_param':
            self.S = np.sqrt(self.r2_act.fibergroup.order())
        elif adjust_channels == 'no_adjust':
            self.S = 1
        else:
            raise ValueError(f"Unsupported adjust_channels option: {adjust_channels}. Only 'keep_channels', 'keep_param', and 'no_adjust' are supported.")

        if maxpool:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (7, 3, 2)
        else:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (3, 1, 1)

        # input type: 3-channel RGB image
        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat_channels = [c * torch_block.expansion for c in [64, 128, 256, 512]]

        self.feat_types = [
            enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * round(c / self.S))
            for c in self.feat_channels
        ]

        # initial conv + BN + ReLU
        self.eq_stages = nn.ModuleList()

        self.conv1 = enn.R2Conv(self.in_type, self.feat_types[0], kernel_size=kernel_s_conv1, stride=stride_s_conv1, padding=padding_s_conv1) # kernel_size=7
        self.eq_stages.append(self.conv1)
        self.bn1 = enn.InnerBatchNorm(self.feat_types[0])
        self.eq_stages.append(self.bn1)
        self.relu = enn.ReLU(self.feat_types[0])
        self.eq_stages.append(self.relu)

        if maxpool:
            self.maxpool = enn.PointwiseMaxPool2D(self.feat_types[0], kernel_size=3, stride=2, padding=1) # kernel_size=3
            self.eq_stages.append(self.maxpool)

        # ResNet layers
        # equivariant blocks
        out_type = self.relu.out_type # initial out_type after conv1, bn1, relu (if eq_blocks=0)
        for i in range(eq_blocks):
            in_type = self.relu.out_type if i == 0 else self.feat_types[i-1]
            out_type = self.feat_types[i]
            stride = 1 if i == 0 else 2
            eq_layer = self._make_layer(eq_block, in_type, out_type, blocks=layers[i], stride=stride)
            self.eq_stages.append(eq_layer)

        # group pooling
        self.gpool = enn.GroupPooling(out_type)
        gpool_channels = self.gpool.out_type.size

        # non-equivariant blocks
        self.torch_stages = nn.ModuleList()
        for i in range(eq_blocks, 4):
            in_channels = gpool_channels if i == eq_blocks else self.feat_channels[i-1]
            out_channels = self.feat_channels[i]
            stride = 1 if i == 0 else 2
            layer = self._make_layer_torch(torch_block, in_channels, out_channels, blocks=layers[i], stride=stride)
            self.torch_stages.append(layer)

        # Pooling (in EqResnet18 vor group pooling -> nn module statt enn !!!!!)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected 
        hidden_dim = round(self.feat_channels[-1] / self.S) if eq_blocks == 4 else self.feat_channels[-1] 
        print(f"Hidden dim before projection head: {hidden_dim}")               
        self.fully_net = nn.Sequential(
            nn.Linear(hidden_dim, projector_hidden_size),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_size, n_classes),
        )

    def _make_layer(self, block, in_type, out_type, blocks, stride=1):
        print('Make equivariant layer')
        layers = []
        downsample = None

        # nach conv downsample fehlt norm layer (enn.InnerBatchNorm)
        if stride != 1 or in_type != out_type:
            downsample = enn.SequentialModule(
                    enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=0, bias=False),# schauen, ob padding benötigt  # conv1x1(in_type, out_type, stride=stride, bias=False)
                    enn.InnerBatchNorm(out_type)
                )
        layers.append(block(in_type, out_type, stride, downsample))

        for _ in range(1, blocks):
            layers.append(block(out_type, out_type))
        
        return enn.SequentialModule(*layers)
    
    def _make_layer_torch(self, block, in_channels, out_channels, blocks, stride=1):
        print('Make non-equivariant layer')
        norm_layer = nn.BatchNorm2d
        downsample = None
        layers = []

        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(out_channels * block.expansion),
            )

        layers.append(block(in_channels, out_channels, stride=stride, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)

        # equivariant 
        for layer in self.eq_stages:
            x = layer(x)

        # group pooling
        x = self.gpool(x)
        x = x.tensor

        # non-equivariant
        for layer in self.torch_stages:
            x = layer(x)
        
        # head
        x = self.avgpool(x)
        hidden = torch.flatten(x, 1)
        z = self.fully_net(hidden)

        return hidden, z

class EqResNet18(nn.Module):
    def __init__(self, N=4, r2_act = None, projector_hidden_size=1024, n_classes=128, gaussian_blur=False, maxpool='max', eq_downsampling=None, adjust_channels=None):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = r2_act if r2_act is not None else gspaces.rot2dOnR2(N)

        self.maxpool = maxpool
        self.eq_downsampling = eq_downsampling
        assert self.eq_downsampling in (None, "kernel_size", "spatial_dim"), \
            f"eq_downsampling must be None, 'kernel_size', or 'spatial_dim', but got: {self.eq_downsampling}"

        if adjust_channels == 'keep_param':
            self.S = np.sqrt(self.r2_act.fibergroup.order())
        elif adjust_channels == 'keep_channels':
            self.S = self.r2_act.fibergroup.order()
        else:
            self.S = 1
        print(f'S = {self.S}')

        if maxpool is None:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (4, 2, 1) if eq_downsampling == "kernel_size" else (3, 1, 1)
            )
        else:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (8, 3, 2) if eq_downsampling == "kernel_size" else (7, 3, 2)
            )
            kernel_s_maxpool = 4 if eq_downsampling == "kernel_size" else 3


        # input type: 3-channel RGB image
        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat64 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(64 / self.S)))
        self.feat128 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(128 / self.S)))
        self.feat256 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(256 / self.S)))
        self.feat512 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(512 / self.S)))

        # initial conv + BN + ReLU
        #self.conv1 = conv7x7(self.in_type, self.feat64, kernel_size=7, stride=2, padding=3)
        if gaussian_blur:
            self.conv1 = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(self.in_type, sigma=0.33, stride=stride_s_conv1, padding=padding_s_conv1), 
                                              enn.R2Conv(self.in_type, self.feat64, kernel_size=kernel_s_conv1, stride=1, padding=3))
        else:
            self.conv1 = enn.R2Conv(self.in_type, self.feat64, kernel_size=kernel_s_conv1, stride=stride_s_conv1, padding=padding_s_conv1) # kernel_size=7

        self.bn1 = enn.InnerBatchNorm(self.feat64)
        self.relu = enn.ReLU(self.feat64)

        if maxpool == 'max':
            self.maxpool = enn.PointwiseMaxPool2D(self.feat64, kernel_size=kernel_s_maxpool, stride=2, padding=1) # kernel_size=3
        elif maxpool == 'avg':
            self.maxpool = enn.PointwiseAvgPoolAntialiased2D(self.feat64, sigma=0.33, stride=2, padding=1)
        else:
            self.maxpool = None

        # ResNet layers
        self.layer1 = self._make_layer(self.relu.out_type, self.feat64, blocks=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer2 = self._make_layer(self.layer1.out_type, self.feat128, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer3 = self._make_layer(self.layer2.out_type, self.feat256, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer4 = self._make_layer(self.layer3.out_type, self.feat512, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        
        # Pooling
        self.avgpool = enn.PointwiseAdaptiveAvgPool(self.layer4.out_type, (1, 1))
        #self.gpool = enn.GroupPooling(self.layer4.out_type) 
        self.gpool = enn.GroupPooling(self.avgpool.out_type)

        # Fully connected
        c = self.gpool.out_type.size
        print('Final feature dimension:', c)
        
        #self.fully_net =  torch.nn.Linear(c, n_classes)
        
        self.fully_net = nn.Sequential(
            nn.Linear(c, projector_hidden_size),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_size, n_classes),
        )

    def _make_layer(self, in_type, out_type, blocks, stride=1, gaussian_blur=False, eq_downsampling=None):
        print('Make layer')
        layers = []
        downsample = None
        if eq_downsampling == 'kernel_size':
            kernel_size = 4
        else:
            kernel_size = 1

        # nach conv downsample fehlt norm layer (enn.InnerBatchNorm)
        if stride != 1 or in_type != out_type:
            if gaussian_blur:
                downsample = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(in_type, sigma=0.33, stride=stride, padding=1), 
                                                  conv1x1(in_type, out_type, stride=1, bias=False))
            else:
                downsample = enn.SequentialModule(
                    enn.R2Conv(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=0, bias=False),# schauen, ob padding benötigt  # conv1x1(in_type, out_type, stride=stride, bias=False)
                    enn.InnerBatchNorm(out_type)
                )
        layers.append(EqBasicBlock(in_type, out_type, stride, downsample, eq_downsampling))
        for _ in range(1, blocks):
            layers.append(EqBasicBlock(out_type, out_type))
        
        return enn.SequentialModule(*layers)
    
    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.gpool(x)

        hidden = self.gpool(x).tensor.squeeze(-2).squeeze(-1)

        # x = x.tensor
        # b, c, w, h = x.shape
        # hidden = F.avg_pool2d(x, (w, h))
        # hidden = hidden.view(hidden.size(0), -1)
        # hidden = torch.flatten(hidden, 1)
        
        z = self.fully_net(hidden)

        return hidden, z

class EqResNet18_hue(nn.Module):
    def __init__(self, N=4, projector_hidden_size=1024, n_classes=128, gaussian_blur=False, maxpool='max', eq_downsampling=None, adjust_channels=None, gpool_mode='max'):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = gspaces.hueOnR2(N)

        self.maxpool = maxpool
        self.eq_downsampling = eq_downsampling
        assert self.eq_downsampling in (None, "kernel_size", "spatial_dim"), \
            f"eq_downsampling must be None, 'kernel_size', or 'spatial_dim', but got: {self.eq_downsampling}"

        if adjust_channels == 'keep_param':
            self.S = np.sqrt(self.r2_act.fibergroup.order())
        elif adjust_channels == 'keep_channels':
            self.S = self.r2_act.fibergroup.order()
        else:
            self.S = 1
        print(f'S = {self.S}')

        if maxpool is None:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (4, 2, 1) if eq_downsampling == "kernel_size" else (3, 1, 1)
            )
        else:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (8, 3, 2) if eq_downsampling == "kernel_size" else (7, 3, 2)
            )
            kernel_s_maxpool = 4 if eq_downsampling == "kernel_size" else 3

        self.gpool_mode = gpool_mode
        self.encoder = enn.HSVHuePhaseEncoder(self.r2_act)

        # input type: 3-channel RGB image
        # self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat64 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(64 / self.S)))
        self.feat128 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(128 / self.S)))
        self.feat256 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(256 / self.S)))
        self.feat512 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(512 / self.S)))

        # initial conv + BN + ReLU
        #self.conv1 = conv7x7(self.in_type, self.feat64, kernel_size=7, stride=2, padding=3)
        if gaussian_blur:
            self.conv1 = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(self.in_type, sigma=0.33, stride=stride_s_conv1, padding=padding_s_conv1), 
                                              enn.R2Conv(self.in_type, self.feat64, kernel_size=kernel_s_conv1, stride=1, padding=3))
        else:
            self.conv1 = enn.R2Conv(self.encoder.out_type, self.feat64, kernel_size=kernel_s_conv1, stride=stride_s_conv1, padding=padding_s_conv1) # kernel_size=7

        self.bn1 = enn.InnerBatchNorm(self.feat64)
        self.relu = enn.ReLU(self.feat64)

        if maxpool == 'max':
            self.maxpool = enn.PointwiseMaxPool2D(self.feat64, kernel_size=kernel_s_maxpool, stride=2, padding=1) # kernel_size=3
        elif maxpool == 'avg':
            self.maxpool = enn.PointwiseAvgPoolAntialiased2D(self.feat64, sigma=0.33, stride=2, padding=1)
        else:
            self.maxpool = None

        # ResNet layers
        self.layer1 = self._make_layer(self.relu.out_type, self.feat64, blocks=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer2 = self._make_layer(self.layer1.out_type, self.feat128, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer3 = self._make_layer(self.layer2.out_type, self.feat256, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer4 = self._make_layer(self.layer3.out_type, self.feat512, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        
        # Pooling
        self.avgpool = enn.PointwiseAdaptiveAvgPool(self.layer4.out_type, (1, 1))
        #self.gpool = enn.GroupPooling(self.layer4.out_type) 
        self.gpool = enn.GroupPooling(self.avgpool.out_type, mode=self.gpool_mode)

        # Fully connected
        c = self.gpool.out_type.size
        print('Final feature dimension:', c)
        
        #self.fully_net =  torch.nn.Linear(c, n_classes)
        
        self.fully_net = nn.Sequential(
            nn.Linear(c, projector_hidden_size),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_size, n_classes),
        )

    def _make_layer(self, in_type, out_type, blocks, stride=1, gaussian_blur=False, eq_downsampling=None):
        print('Make layer')
        layers = []
        downsample = None
        if eq_downsampling == 'kernel_size':
            kernel_size = 4
        else:
            kernel_size = 1

        # nach conv downsample fehlt norm layer (enn.InnerBatchNorm)
        if stride != 1 or in_type != out_type:
            if gaussian_blur:
                downsample = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(in_type, sigma=0.33, stride=stride, padding=1), 
                                                  conv1x1(in_type, out_type, stride=1, bias=False))
            else:
                downsample = enn.SequentialModule(
                    enn.R2Conv(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=0, bias=False),# schauen, ob padding benötigt  # conv1x1(in_type, out_type, stride=stride, bias=False)
                    enn.InnerBatchNorm(out_type)
                )
        layers.append(EqBasicBlock(in_type, out_type, stride, downsample, eq_downsampling))
        for _ in range(1, blocks):
            layers.append(EqBasicBlock(out_type, out_type))
        
        return enn.SequentialModule(*layers)
    
    def forward(self, x):
        x = self.encoder(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.gpool(x)
        #print('Shape after avgpool:', x.tensor.shape)

        hidden = self.gpool(x).tensor.squeeze(-2).squeeze(-1)
        #hidden = x.tensor.mean((-2,-1))
        #print('Hidden representation shape:', hidden.shape)
        # x = x.tensor
        # b, c, w, h = x.shape
        # hidden = F.avg_pool2d(x, (w, h))
        # hidden = hidden.view(hidden.size(0), -1)
        # hidden = torch.flatten(hidden, 1)
        
        z = self.fully_net(hidden)

        return hidden, z

class EqResNet18_hue2(nn.Module):
    def __init__(self, N=4, projector_hidden_size=1024, n_classes=128, gaussian_blur=False, maxpool='max', eq_downsampling=None, adjust_channels=None, gpool_mode='max'):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = gspaces.hueOnR2(N)

        self.maxpool = maxpool
        self.eq_downsampling = eq_downsampling
        assert self.eq_downsampling in (None, "kernel_size", "spatial_dim"), \
            f"eq_downsampling must be None, 'kernel_size', or 'spatial_dim', but got: {self.eq_downsampling}"

        if adjust_channels == 'keep_param':
            self.S = np.sqrt(self.r2_act.fibergroup.order())
        elif adjust_channels == 'keep_channels':
            self.S = self.r2_act.fibergroup.order()
        else:
            self.S = 1
        print(f'S = {self.S}')

        if maxpool is None:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (4, 2, 1) if eq_downsampling == "kernel_size" else (3, 1, 1)
            )
        else:
            kernel_s_conv1, padding_s_conv1, stride_s_conv1 = (
                (8, 3, 2) if eq_downsampling == "kernel_size" else (7, 3, 2)
            )
            kernel_s_maxpool = 4 if eq_downsampling == "kernel_size" else 3

        self.gpool_mode = gpool_mode
        self.encoder = enn.HSVHuePhaseEncoder(self.r2_act)

        # input type: 3-channel RGB image
        # self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat64 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(64 / self.S)))
        self.feat128 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(128 / self.S)))
        self.feat256 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(256 / self.S)))
        self.feat512 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * (round(512 / self.S)))

        # initial conv + BN + ReLU
        #self.conv1 = conv7x7(self.in_type, self.feat64, kernel_size=7, stride=2, padding=3)
        if gaussian_blur:
            self.conv1 = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(self.in_type, sigma=0.33, stride=stride_s_conv1, padding=padding_s_conv1), 
                                              enn.R2Conv(self.in_type, self.feat64, kernel_size=kernel_s_conv1, stride=1, padding=3))
        else:
            self.conv1 = enn.R2Conv(self.encoder.out_type, self.feat64, kernel_size=kernel_s_conv1, stride=stride_s_conv1, padding=padding_s_conv1) # kernel_size=7

        self.bn1 = enn.InnerBatchNorm(self.feat64)
        self.relu = enn.ReLU(self.feat64)

        if maxpool == 'max':
            self.maxpool = enn.PointwiseMaxPool2D(self.feat64, kernel_size=kernel_s_maxpool, stride=2, padding=1) # kernel_size=3
        elif maxpool == 'avg':
            self.maxpool = enn.PointwiseAvgPoolAntialiased2D(self.feat64, sigma=0.33, stride=2, padding=1)
        else:
            self.maxpool = None

        # ResNet layers
        self.layer1 = self._make_layer(self.relu.out_type, self.feat64, blocks=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer2 = self._make_layer(self.layer1.out_type, self.feat128, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer3 = self._make_layer(self.layer2.out_type, self.feat256, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        self.layer4 = self._make_layer(self.layer3.out_type, self.feat512, blocks=2, stride=2, gaussian_blur=gaussian_blur, eq_downsampling=eq_downsampling)
        
        # Pooling
        self.avgpool = enn.PointwiseAdaptiveAvgPool(self.layer4.out_type, (1, 1))
        #self.gpool = enn.GroupPooling(self.layer4.out_type) 
        self.gpool = enn.GroupPooling(self.avgpool.out_type, mode=self.gpool_mode)

        # Fully connected
        c = self.gpool.out_type.size
        print('Final feature dimension:', c)
        
        #self.fully_net =  torch.nn.Linear(c, n_classes)
        
        self.fully_net = nn.Sequential(
            nn.Linear(c, projector_hidden_size),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(projector_hidden_size, n_classes),
        )

    def _make_layer(self, in_type, out_type, blocks, stride=1, gaussian_blur=False, eq_downsampling=None):
        print('Make layer2')
        layers = []
        downsample = None
        if eq_downsampling == 'kernel_size':
            kernel_size = 4
        else:
            kernel_size = 1

        # nach conv downsample fehlt norm layer (enn.InnerBatchNorm)
        if stride != 1 or in_type != out_type:
            if gaussian_blur:
                downsample = enn.SequentialModule(enn.PointwiseAvgPoolAntialiased2D(in_type, sigma=0.33, stride=stride, padding=1), 
                                                  conv1x1(in_type, out_type, stride=1, bias=False))
            else:
                downsample = enn.SequentialModule(
                    enn.R2HueGroupConv(in_type, out_type, kernel_size=kernel_size, stride=stride, padding=0, bias=False),# schauen, ob padding benötigt  # conv1x1(in_type, out_type, stride=stride, bias=False)
                    enn.InnerBatchNorm(out_type)
                )
        layers.append(EqBasicBlock_hue(in_type, out_type, stride, downsample, eq_downsampling))
        for _ in range(1, blocks):
            layers.append(EqBasicBlock_hue(out_type, out_type))
        
        return enn.SequentialModule(*layers)
    
    def forward(self, x):
        x = self.encoder(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.gpool(x)
        #print('Shape after avgpool:', x.tensor.shape)

        hidden = self.gpool(x).tensor.squeeze(-2).squeeze(-1)
        #hidden = x.tensor.mean((-2,-1))
        #print('Hidden representation shape:', hidden.shape)
        # x = x.tensor
        # b, c, w, h = x.shape
        # hidden = F.avg_pool2d(x, (w, h))
        # hidden = hidden.view(hidden.size(0), -1)
        # hidden = torch.flatten(hidden, 1)
        
        z = self.fully_net(hidden)

        return hidden, z
    
class eqEncoder_Resnet(nn.Module):
    def __init__(self, backbone_network=resnet18, N=16, encoder_channels=16, projector_hidden_size=1024, n_classes=128, maxpool=True):
        super().__init__()
        # Define the rotational and flip symmetry group
        self.r2_act = gspaces.rot2dOnR2(N)

        self.in_type = enn.FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])

        # feature types for each stage
        self.feat64 = enn.FieldType(self.r2_act, [self.r2_act.regular_repr] * encoder_channels)

        # Equivariant block
        self.conv1 = enn.R2Conv(self.in_type, self.feat64, kernel_size=3, stride=1, padding=1)
        self.bn1 = enn.InnerBatchNorm(self.feat64)
        self.relu = enn.ReLU(self.feat64)
        self.gpool = enn.GroupPooling(self.relu.out_type) # make invariant
        c = self.gpool.out_type.size
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # helpful?

        # make normal first layers identity/disappear
        self.backbone = backbone_network(weights=None)
        self.backbone_output_dim = self.backbone.fc.in_features

        if maxpool:
            self.backbone.conv1 = nn.Conv2d(
                c, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.backbone.conv1 = nn.Conv2d(
                c, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.backbone.maxpool = nn.Identity()
        
        
        self.backbone.fc = nn.Identity()

        # projection head
        self.projector = nn.Sequential(
            nn.Linear(self.backbone_output_dim, projector_hidden_size), 
            nn.ReLU(), 
            nn.Linear(projector_hidden_size, n_classes),
        )
    
    def forward(self, x):
        # equivariant block
        x = enn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.gpool(x)
        x = x.tensor

        # backbone and projector
        h = self.backbone(x)
        z = self.projector(h)
        return h, z

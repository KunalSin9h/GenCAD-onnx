# Imports
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
from tqdm import tqdm 
import torchvision
from torchvision.transforms import transforms
from utils.cad_dataset import get_ccip_dataloader
from config import ConfigCCIP




from typing import Callable, List, Optional, Type

import torch.nn as nn
from torch import Tensor



from typing import Callable, List, Optional, Type

import torch.nn as nn
from torch import Tensor

"""From https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18"""

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockEnc(nn.Module):
    """The basic block architecture of resnet-18 network.
    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        DROPOUT = 0.1

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Encoder(nn.Module):
    """The encoder model.
    """
    def __init__(
        self,
        block: Type[BasicBlockEnc],
        layers: List[int],
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlockEnc) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlockEnc],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)



class ResNetImageEncoder(nn.Module):
    """Construction of resnet autoencoder.

    Attributes:
        network (str): the architectural type of the network. There are 2 choices:
            - 'default' (default), related with the original resnet-18 architecture
            - 'light', a samller network implementation of resnet-18 for smaller input images.
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types 
            of network, 34 layers for default only network and 20 layers for light network. 
    """

    def __init__(self, network='resnet-18'):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for 
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()
        self.network = network
        self.tanh = nn.Tanh()
        
        if self.network == 'resnet-10':
            # resnet 10 encoder
            self.encoder = Encoder(BasicBlockEnc, [1, 1, 1, 1]) 
        elif self.network == 'resnet-18':
            # resnet 18 encoder
            self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2]) 
        elif self.network == 'resnet-34':
            # resnet 34 encoder
            self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3]) 
        else:
            raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
            
        # self.tanh = nn.Tanh()

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): the batched input data

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        z = self.encoder(x)  # (B, 512, 8, 8)
        z = z.view(z.size(0), -1)  # (B, 512 * 8 * 8)
        return z






class MultiViewResNetImageEncoder(nn.Module):
    """Construction of resnet autoencoder.

    Attributes:
        network (str): the architectural type of the network. There are 2 choices:
            - 'default' (default), related with the original resnet-18 architecture
            - 'light', a samller network implementation of resnet-18 for smaller input images.
        num_layers (int): the number of layers to be created. Implemented for 18 layers (default) for both types 
            of network, 34 layers for default only network and 20 layers for light network. 
    """

    def __init__(self, network='resnet-18'):
        """Initialize the autoencoder.

        Args:
            network (str): a flag to efine the network version. Choices ['default' (default), 'light'].
             num_layers (int): the number of layers to be created. Choices [18 (default), 34 (only for 
                'default' network), 20 (only for 'light' network).
        """
        super().__init__()
        self.network = network
        self.tanh = nn.Tanh()
        
        if self.network == 'resnet-10':
            # resnet 10 encoder
            self.encoder = Encoder(BasicBlockEnc, [1, 1, 1, 1]) 
        elif self.network == 'resnet-18':
            # resnet 18 encoder
            self.encoder = Encoder(BasicBlockEnc, [2, 2, 2, 2]) 
        elif self.network == 'resnet-34':
            # resnet 34 encoder
            self.encoder = Encoder(BasicBlockEnc, [3, 4, 6, 3]) 
        else:
            raise NotImplementedError("Only resnet 18 & 34 autoencoder have been implemented for images size >= 64x64.")
            
        # self.tanh = nn.Tanh()

    def forward(self, x):
        """The forward functon of the model.

        Args:
            x (torch.tensor): (B, 15, 256, 256)

        Returns:
            x (torch.tensor): encoder result
            z (torch.tensor): decoder result
        """
        B, C, H, W = x.shape  # (B, 15, 256, 256)
        slices = C // 3
        z_list = []

        for i in range(slices): 
            x_slice = x[:, i*3:i*3+3, :, :]  # (B, 3, 256, 256)
            z = self.encoder(x_slice)  # (B, 512, 8, 8)
            z = z.view(z.shape[0], -1).unsqueeze(1)
            z_list.append(z)

        z_concat = torch.cat(z_list, dim=1)  # (B, 5, 512 * 8 * 8)

        return z_concat

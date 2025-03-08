import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

__all__ = ['SpikingResNet', 'spiking_resnet18', 'spiking_resnet34', 'spiking_resnet50', 'spiking_resnet101',
           'spiking_resnet152']



def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=mid_channels, 
                kernel_size=3, 
                stride=stride, 
                padding=1, 
                bias=False
            ),
            norm_layer(mid_channels)
        )
        self.sn1 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=mid_channels, 
                out_channels=mid_channels, 
                kernel_size=3, 
                stride=1, 
                padding=1, 
                bias=False
                ),
            norm_layer(mid_channels)
        )
        self.sn2 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.sn2(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, mid_channels, stride=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                bias=False,
                ),
            norm_layer(mid_channels)
        )
        self.sn1 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.conv2 = layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                ),
            norm_layer(mid_channels)
        )
        self.sn2 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.conv3 = layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels=mid_channels, 
                out_channels=mid_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            norm_layer(mid_channels * self.expansion)
        )

        self.sn3 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.sn3(out)


class SpikingResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, norm_layer=None, T=8):
        super(SpikingResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_channels = 64

        self.conv1 = layer.SeqToANNContainer(
            nn.Conv2d(in_channels=2, out_channels=self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(self.in_channels)
            )
        self.avgpool1 = layer.SeqToANNContainer(
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        self.sn1 = neuron.IFNode(step_mode='m', detach_reset=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool2 = layer.SeqToANNContainer(
            nn.AdaptiveAvgPool2d((1, 1))
            )
        self.fc =layer.SeqToANNContainer(
            nn.Linear(512 * block.expansion, num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, mid_channels, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.in_channels != mid_channels * block.expansion:  # downsample
            downsample = layer.SeqToANNContainer(
                    nn.Conv2d(
                        in_channels=self.in_channels,
                        out_channels=mid_channels * block.expansion,
                        kernel_size=1, 
                        stride=stride
                        ),
                    norm_layer(mid_channels * block.expansion),
                )
        layers = []
        layers.append(
            block(
                in_channels=self.in_channels,
                mid_channels=mid_channels,
                stride=stride,
                downsample=downsample,
                norm_layer=norm_layer
                )
            )
        
        self.in_channels = mid_channels * block.expansion  # update in_channels
        
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=self.in_channels, 
                    mid_channels=mid_channels,
                    stride=1,
                    downsample=None, 
                    norm_layer=norm_layer
                    )
                )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.avgpool1(x)
        x = self.sn1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool2(x)
        x = torch.flatten(x, 2)
        self.feature = x
        x = self.fc(x)
        return x.mean(dim=0)    # x.shape = (N, C)

    def forward(self, x):
        return self._forward_impl(x)


def _spiking_resnet(block, layers, **kwargs):
    model = SpikingResNet(block, layers, **kwargs)
    return model


def spiking_resnet18(**kwargs):
    return _spiking_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def spiking_resnet34(**kwargs):
    return _spiking_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def spiking_resnet50(**kwargs):
    return _spiking_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def spiking_resnet101(**kwargs):
    return _spiking_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def spiking_resnet152(**kwargs):
    return _spiking_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)





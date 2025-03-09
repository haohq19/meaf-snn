import torch.nn as nn
from spikingjelly.activation_based import layer, neuron

__all__ = ['SpikingMLP', 'spiking_mlp12', 'spiking_mlp6']


class SpikingMLP(nn.Module):

    def __init__(self, cfg_layers, num_classes=1000, T=8):
        super(SpikingMLP, self).__init__()
        self.T = T
        self.cfg_layers = cfg_layers # list, number of neurons in each layer
        self.avgpool = nn.AdaptiveAvgPool2d((128, 128))
        self.sn0 = neuron.IFNode(step_mode='m', detach_reset=True)
        
        self.layers = []
        in_features = 128 * 128 * 2
        for out_features in cfg_layers:
            self.layers.append(self._make_layer(in_features, out_features))
            in_features = out_features

        self.layers = nn.Sequential(*self.layers)

        self.output =layer.SeqToANNContainer(
            nn.Linear(cfg_layers[-1], num_classes)
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, in_features, out_features):
        fc = layer.SeqToANNContainer(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
        )   
        sn = neuron.IFNode(step_mode='m', detach_reset=True)
        mlp_layer = nn.Sequential(fc, sn)
        return mlp_layer

    def _forward_impl(self, x):
        # x.shape = (T, B, C, H, W)
        T, B, C, H, W = x.size()
        if H < 128 or W < 128:
            x = x.reshape(T * B, C, H, W)
            x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)   
        elif H > 128 or W > 128:
            x = x.reshape(T * B, C, H, W)
            x = self.avgpool(x)
        x = x.reshape(T, B, -1)
        x = self.sn0(x)

        for layer in self.layers:
            x = layer(x)
        
        self.feature = x

        x = self.output(x)
        return x.mean(dim=0)    # x.shape = (B, C)

    def forward(self, x):
        return self._forward_impl(x)


def _spiking_mlp(num_layers, **kwargs):
    model = SpikingMLP(num_layers, **kwargs)
    return model


def spiking_mlp12(**kwargs):
    cfg_layers = [2048, 2048, 1024, 1024, 1024, 1024, 512, 512, 512, 512, 512, 512]
    return _spiking_mlp(cfg_layers, **kwargs)

def spiking_mlp6(**kwargs):
    cfg_layers = [2048, 1024, 1024, 512, 512, 512]
    return _spiking_mlp(cfg_layers, **kwargs)





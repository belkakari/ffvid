from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianFourierFeatureTransform(nn.Module):
    """
    From https://github.com/ndahlquist/pytorch-fourier-feature-networks/blob/master/fourier_feature_transform.py
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, num_input_channels, width, height],
     returns a tensor of size [batches, mapping_dim*2, width, height].
    """

    def __init__(self, num_input_channels=2, mapping_dim=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self.mapping_dim = mapping_dim
        self._B = torch.randn((num_input_channels, mapping_dim)) * scale

    def forward(self, x, phase=None):
        assert x.dim() == 4, 'Expected 4D input (got {}D input)'.format(x.dim())

        batches, channels, width, height = x.shape
        assert channels == self._num_input_channels,\
            "Expected input to have {} channels (got {} channels)".format(self._num_input_channels, channels)

        # Make shape compatible for matmul with _B.
        # From [B, C, W, H] to [(B*W*H), C].
        x = x.permute(0, 2, 3, 1).reshape(batches * width * height, channels)

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = x.view(batches, width, height, self.mapping_dim)
        # From [B, W, H, C] to [B, C, W, H]
        x = x.permute(0, 3, 1, 2)

        if phase is not None:
            x = 2 * pi * x + phase
        else:
            x = 2 * pi * x
        
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)

class LFF(nn.Module):
    """
    From https://github.com/saic-mdal/CIPS/blob/main/model/blocks.py
    Learnable fourier features
    """
    def __init__(self, mapping_dim, num_input_channels=2):
        super(LFF, self).__init__()
        self.ffm = ConLinear(num_input_channels, mapping_dim, is_first=True)
        self.activation = SinActivation()

    def forward(self, x, phase=None):            
        x = self.ffm(x)
        if phase is not None:
            x = self.activation(x + phase)
        else:
            x = self.activation(x)
        return x
    
class ConLinear(nn.Module):
    """
    From https://github.com/saic-mdal/CIPS/blob/main/model/blocks.py
    Linear 1x1 convolution with weight initialization suitable for sine activation
    """
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super(ConLinear, self).__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)
        if is_first:
            nn.init.uniform_(self.conv.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.conv.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))

    def forward(self, x):
        return self.conv(x)


class PixelwiseConv(nn.Module):
    def __init__(self, ch_in, ch_out, bias=True, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(x)


class SinActivation(nn.Module):
    """
    From https://github.com/saic-mdal/CIPS/blob/main/model/blocks.py
    Sine activation function with phase modulation
    """
    def __init__(self,):
        super(SinActivation, self).__init__()

    def forward(self, x, phase=None):
        if phase:
            return torch.sin(x + phase)
        else:
            return torch.sin(x)

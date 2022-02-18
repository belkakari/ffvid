import torch
import torch.nn as nn

from ffvid.modules import ConLinear, LFF, GaussianFourierFeatureTransform


class FMLP(nn.Module):
    def __init__(self,
                 internal_dim,
                 num_layers=3,
                 act=nn.LeakyReLU(), 
                 ff_func=LFF,
                 phase_mod_type=None,
                 num_input_channels=2,
                 conv=ConLinear,
                ):
        super().__init__()

        if ff_func is GaussianFourierFeatureTransform:
            self.mult = 2  # gaussian mapping concatenates sin and cos, effectively doubling the channels
        else:
            self.mult = 1

        self.phase_mod_type = phase_mod_type

        self.lff = ff_func(mapping_dim=internal_dim, num_input_channels=num_input_channels)
        self.net = [conv(internal_dim * self.mult, internal_dim * 2, is_first=False),
                    act,
        ]
        for layer_n in range(num_layers):
            self.net.append(conv(internal_dim * 2, internal_dim * 2, is_first=False))
            self.net.append(act)
    
        self.net.append(conv(internal_dim * 2, 3, is_first=False))
        self.net = nn.Sequential(*self.net)
        
        if phase_mod_type == 'pixelwise':
            self.phase_net = [ff_func(mapping_dim=internal_dim // self.mult,
                                      num_input_channels=3),
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        1,
                                        is_first=False),]
            self.phase_net = nn.Sequential(*self.phase_net)
        
        if phase_mod_type == 'freqwise':
            self.phase_net = [ff_func(mapping_dim=internal_dim // self.mult,
                                      num_input_channels=1),
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),
                              act,
                              conv(internal_dim,
                                        internal_dim,
                                        is_first=False),]
            self.phase_net = nn.Sequential(*self.phase_net)

    def forward(self, coords):
        if self.phase_mod_type == 'pixelwise':
            # coords: list of [spatial_coords:[B,2,H,W], spacetime_coords[B, 3, H, W]]
            spatial_coords, time_coords = coords
            phase_feats = self.phase_net(time_coords)
            fourier_feats = self.lff(spatial_coords, phase_feats)
        if self.phase_mod_type == 'freqwise':
            # coords: list of [spatial_coords:[B,2,H,W], spacetime_coords[B, 3, H, W]]
            spatial_coords, time_coords = coords
            phase_feats = self.phase_net(time_coords[:, [2]].mean(2, keepdims=True).mean(3, keepdims=True))
            fourier_feats = self.lff(spatial_coords, phase_feats.repeat(1, 1, *spatial_coords.shape[-2:]))
        else:
            fourier_feats = self.lff(coords)
        return self.net(fourier_feats)


class PhaseMLP(nn.Module):
    def __init__(self,
                 num_input_channels,
                 internal_dim,
                 num_layers,
                 num_freqs,
                 ff_func=GaussianFourierFeatureTransform,
                 conv=ConLinear,
                 act=nn.LeakyReLU(),
                 *args,
                 **kwargs,
                 ):
        super().__init__()
        if ff_func is GaussianFourierFeatureTransform:
            self.mult = 2  # gaussian mapping concatenates sin and cos, effectively doubling the channels
        else:
            self.mult = 1

        self.ff = ff_func(mapping_dim=num_freqs // self.mult, 
                           num_input_channels=num_input_channels)
        self.net = [conv(num_freqs, internal_dim * 2, is_first=False),
                    act,
        ]
        for layer_n in range(num_layers):
            self.net.append(conv(internal_dim * 2, internal_dim * 2, is_first=False))
            self.net.append(act)
    
        self.net.append(conv(internal_dim * 2, num_freqs, is_first=False))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(self.ff(coords))


class CoordMLP(nn.Module):
    def __init__(self,
                 internal_dim,
                 num_layers=3,
                 act=nn.LeakyReLU(), 
                 ff_func=LFF,
                 num_input_channels=2,
                 conv=ConLinear,
                 num_freqs=30,
                 *args,
                 **kwargs,):
        super().__init__()
        if ff_func is GaussianFourierFeatureTransform:
            self.mult = 2  # gaussian mapping concatenates sin and cos, effectively doubling the channels
        else:
            self.mult = 1

        self.lff = ff_func(mapping_dim=num_freqs, num_input_channels=num_input_channels)
        self.net = [conv(num_freqs * self.mult, internal_dim * 2, is_first=False),
                    act,
        ]
        for layer_n in range(num_layers):
            self.net.append(conv(internal_dim * 2, internal_dim * 2, is_first=False))
            self.net.append(act)
    
        self.net.append(conv(internal_dim * 2, 3, is_first=False))
        self.net = nn.Sequential(*self.net)

    def forward(self, coords, phase=None):
        fourier_feats = self.lff(coords, phase)
        return self.net(fourier_feats)


class ModulatedMLP(nn.Module):
    def __init__(self, 
                 renderrer,
                 modulator,
                 *args,
                 **kwargs,):
        super().__init__()
        self.renderrer = renderrer
        self.modulator = modulator
    
    def forward(self, coords):
        spatial_coords, time_coords = coords
        phase_feats = self.modulator(time_coords[:, [2]].mean(2, keepdims=True).mean(3, keepdims=True))
        return self.renderrer(spatial_coords, phase_feats.repeat(1, 1, *spatial_coords.shape[-2:]))

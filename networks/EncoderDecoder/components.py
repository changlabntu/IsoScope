# models/gan/components.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizationFactory:
    """Factory class for creating normalization layers."""

    @staticmethod
    def get_norm(out_channels, method, dim='3d'):
        if method == 'batch':
            if dim == '1d':
                return nn.BatchNorm1d(out_channels)
            elif dim == '2d':
                return nn.BatchNorm2d(out_channels)
            elif dim == '3d':
                return nn.BatchNorm3d(out_channels)
        elif method == 'instance':
            if dim == '1d':
                return nn.InstanceNorm1d(out_channels)
            elif dim == '2d':
                return nn.InstanceNorm2d(out_channels)
            elif dim == '3d':
                return nn.InstanceNorm3d(out_channels)
        elif method == 'group':
            return nn.GroupNorm(32, out_channels)
        return nn.Identity()


class ConvBlock(nn.Module):
    """Configurable convolution block with normalization."""

    def __init__(self, in_channels, out_channels, dim='3d', kernel=3,
                 activation=nn.ReLU, norm='batch'):
        super().__init__()
        self.dim = dim
        Conv = {
            '1d': nn.Conv1d,
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }[dim]

        self.conv = Conv(in_channels, out_channels, kernel, padding=1)
        self.norm = NormalizationFactory.get_norm(out_channels, norm, dim)
        self.activation = activation()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


class DeconvBlock(nn.Module):
    """Configurable deconvolution block with normalization."""

    def __init__(self, in_channels, out_channels, dim='3d',
                 use_upsample=(2, 2, 2), activation=nn.ReLU, norm='batch'):
        super().__init__()

        Conv = {
            '1d': nn.Conv1d,
            '2d': nn.Conv2d,
            '3d': nn.Conv3d
        }[dim]

        self.up = nn.Upsample(scale_factor=use_upsample)
        self.conv = Conv(in_channels, out_channels, 3, padding=1)
        self.norm = NormalizationFactory.get_norm(out_channels, norm, dim)
        self.activation = activation()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.norm(x)
        return self.activation(x)


# models/gan/base.py
class BaseGenerator(nn.Module):
    """Base generator class defining common interface."""

    def __init__(self, n_channels=1, out_channels=1, nf=32,
                 norm_type='batch', activation=nn.ReLU,
                 final='tanh', mc=False):
        super().__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.nf = nf
        self.norm_type = norm_type
        self.activation = activation
        self.dropout = 0.5 if mc else 0.0
        self.final_activation = self._get_final_activation(final)

    def _get_final_activation(self, final):
        if final == 'tanh':
            return nn.Tanh()
        elif final == 'sigmoid':
            return nn.Sigmoid()
        return nn.Identity()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, features):
        raise NotImplementedError

    def forward(self, x, method=None):
        if method == 'encode':
            return self.encode(x)
        elif method == 'decode':
            return self.decode(x)

        features = self.encode(x)
        return self.decode(features)


# models/gan/variants/clean.py
class CleanGenerator(BaseGenerator):
    """Clean version of the generator with direct 3D processing."""

    def __init__(self, *args, encode='3d', decode='3d', **kwargs):
        super().__init__(*args, **kwargs)
        self.encode_dim = encode
        self.decode_dim = decode

        # Build encoder blocks
        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        # Output heads
        self.conv7_k = self._build_output_head()
        self.conv7_g = self._build_output_head()

    def _build_encoder(self):
        # Implementation of encoder blocks
        blocks = []
        channels = [self.n_channels, self.nf, 2 * self.nf, 4 * self.nf, 8 * self.nf]

        for i in range(len(channels) - 1):
            block = nn.Sequential(
                ConvBlock(channels[i], channels[i + 1],
                          dim=self.encode_dim,
                          activation=self.activation,
                          norm=self.norm_type),
                ConvBlock(channels[i + 1], channels[i + 1],
                          dim=self.encode_dim,
                          activation=self.activation,
                          norm=self.norm_type)
            )
            if i > 1:
                block.insert(1, nn.Dropout(self.dropout))
            blocks.append(block)

        return nn.ModuleList(blocks)

    def _build_decoder(self):
        # Implementation of decoder blocks
        return nn.ModuleList([
            DeconvBlock(8 * self.nf, 4 * self.nf, dim=self.decode_dim),
            ConvBlock(8 * self.nf, 4 * self.nf, dim=self.decode_dim),
            DeconvBlock(4 * self.nf, 2 * self.nf, dim=self.decode_dim),
            ConvBlock(4 * self.nf, 2 * self.nf, dim=self.decode_dim),
            DeconvBlock(2 * self.nf, self.nf, dim=self.decode_dim)
        ])

    def _build_output_head(self):
        return ConvBlock(2 * self.nf, self.out_channels,
                         dim=self.decode_dim,
                         activation=lambda: self.final_activation,
                         norm='none')

    def encode(self, x):
        features = []
        for i, block in enumerate(self.encoder):
            if i > 0:
                x = nn.functional.max_pool3d(x, 2)
            x = block(x)
            features.append(x)
        return features

    def decode(self, features):
        [x0, x1, x2, x3] = features

        # Decoder path with skip connections
        xu3 = self.decoder[0](x3)
        cat3 = torch.cat([xu3, x2], 1)
        x5 = self.decoder[1](cat3)

        xu2 = self.decoder[2](x5)
        cat2 = torch.cat([xu2, x1], 1)
        x6 = self.decoder[3](cat2)

        xu1 = self.decoder[4](x6)
        cat1 = torch.cat([xu1, x0], 1)

        return {
            'out0': self.conv7_k(cat1),
            'out1': self.conv7_g(cat1)
        }


# models/gan/factory.py
class GeneratorFactory:
    """Factory for creating different generator variants."""

    @staticmethod
    def create(variant='clean', **kwargs):
        variants = {
            'clean': CleanGenerator,
            # Add other variants here
        }

        if variant not in variants:
            raise ValueError(f"Unknown variant: {variant}")

        return variants[variant](**kwargs)


# Example usage
def create_model(variant='clean', **kwargs):
    return GeneratorFactory.create(variant, **kwargs)
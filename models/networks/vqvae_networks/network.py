# adopt from: 
# - VQVAE: https://github.com/nadavbh12/VQ-VAE
# - Encoder: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py

from __future__ import print_function

import torch
import torch.utils.data
from torch import nn
from torch.nn import init
from torch.nn import functional as F

from einops import rearrange

from models.networks.vqvae_networks.vqvae_modules import Encoder3D, Decoder3D
from models.networks.vqvae_networks.quantizer import VectorQuantizer

def init_weights(net, init_type='normal', gain=0.01):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

    # propagate to children
    for m in net.children():
        m.apply(init_func)


class VQVAE(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 ):
        super(VQVAE, self).__init__()

        self.ddconfig = ddconfig
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.encoder = Encoder3D(**ddconfig)
        self.decoder = Decoder3D(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=1.0,
                                        remap=remap, sane_index_shape=sane_index_shape, legacy=False)
        self.quant_conv = torch.nn.Conv3d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(embed_dim, ddconfig["z_channels"], 1)

        init_weights(self.encoder, 'normal', 0.02)
        init_weights(self.decoder, 'normal', 0.02)
        init_weights(self.quant_conv, 'normal', 0.02)
        init_weights(self.post_quant_conv, 'normal', 0.02)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return quant, emb_loss, info

    def encode_no_quant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        # quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_no_quant(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h, is_voxel=True)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_from_quant(self,quant_code):
        embed_from_code = self.quantize.embedding(quant_code)
        return embed_from_code
    
    def decode_enc_idices(self, enc_indices, z_spatial_dim=8):

        # for transformer
        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices) # (bs t) zd
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3', d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, verbose=False, forward_no_quant=False, encode_only=False):

        if forward_no_quant:
            # for diffusion model's training
            z = self.encode_no_quant(input)
            if encode_only:
                return z
                
            dec = self.decode_no_quant(z)
            return dec, z

        quant, diff, info = self.encode(input)
        dec = self.decode(quant)

        if verbose:
            return dec, quant, diff, info
        else:
            return dec, diff

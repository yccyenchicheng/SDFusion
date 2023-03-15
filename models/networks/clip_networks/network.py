""" 
    Reference:
        - https://github.com/CompVis/latent-diffusion/blob/main/ldm/modules/encoders/modules.py
        - https://github.com/openai/CLIP
"""

import kornia
from einops import rearrange, repeat

import torch
import torch.nn as nn

from external.clip import clip

class CLIPImageEncoder(nn.Module):
    def __init__(
            self,
            model="ViT-B/32",
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
        ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        # self.model, self.preprocess = clip.load(name=model, device=device, jit=jit)
        self.model = self.model.float() # turns out this is important...

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic',align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))

# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
import os
from collections import OrderedDict
from functools import partial

import cv2
import numpy as np
import einops
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.vqvae_networks.network import VQVAE
from models.networks.diffusion_networks.network import DiffusionUNet
from models.networks.resnet_v1 import resnet18
from models.networks.bert_networks.network import BERTTextEncoder
from models.model_utils import load_vqvae

## ldm util
from models.networks.diffusion_networks.ldm_diffusion_util import (
    make_beta_schedule,
    extract_into_tensor,
    noise_like,
    exists,
    default,
)
from models.networks.diffusion_networks.samplers.ddim import DDIMSampler

# distributed 
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf

class SDFusionMultiModal2ShapeModel(BaseModel):
    def name(self):
        return 'SDFusion-Multi-Modal-Conditional-Shape-Generation-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device

        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        # record z_shape
        ddconfig = vq_conf.model.params.ddconfig
        shape_res = ddconfig.resolution
        z_ch, n_down = ddconfig.z_channels, len(ddconfig.ch_mult)-1
        z_sp_dim = shape_res // (2 ** n_down)
        self.z_shape = (z_ch, z_sp_dim, z_sp_dim, z_sp_dim)

        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.uc_scale = 1.
        self.df = DiffusionUNet(unet_params, vq_conf=vq_conf, conditioning_key=df_model_params.conditioning_key)
        self.df.to(self.device)
        self.init_diffusion_params(uc_scale=self.uc_scale, opt=opt)
        
        # sampler 
        self.ddim_sampler = DDIMSampler(self)

        # init vqvae
        self.vqvae = load_vqvae(vq_conf, vq_ckpt=opt.vq_ckpt, opt=opt)

        # init cond model
        self.img_enc = resnet18(pretrained=True) # context dim: 512
        self.img_enc.to(self.device)
        for param in self.img_enc.parameters():
            param.requires_grad = True
            
        # map to text_dim
        img_context_d = 512
        txt_context_d = df_conf.bert.params.n_embed
        self.img_linear = nn.Linear(img_context_d, txt_context_d)
        self.img_linear.to(self.device)
            
        # bert
        bert_params = df_conf.bert.params
        self.text_embed_dim = bert_params.n_embed
        self.txt_enc = BERTTextEncoder(**bert_params)
        self.txt_enc.to(self.device)
        for param in self.txt_enc.parameters():
            param.requires_grad = True
        ######## END: Define Networks ########

        # param list
        trainable_models = [self.df, self.img_enc, self.img_linear, self.txt_enc]
        trainable_params = []

        for m in trainable_models:
            trainable_params += [p for p in m.parameters() if p.requires_grad == True]

        if self.isTrain:
            # initialize optimizers
            self.optimizer = optim.AdamW(trainable_params, lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            
        # transforms
        self.to_tensor = transforms.ToTensor()

        # setup renderer
        dist, elev, azim = 1.7, 20, 20   
        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.opt.device)

        # for multi-gpu
        if self.opt.distributed:
            self.make_distributed(opt)

            self.df_module = self.df.module
            self.vqvae_module = self.vqvae.module
            self.img_enc_module = self.img_enc.module
            self.img_linear_module = self.img_linear.module
            self.txt_enc_module = self.txt_enc.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae
            self.img_enc_module = self.img_enc
            self.img_linear_module = self.img_linear
            self.txt_enc_module = self.txt_enc

        # for debugging purpose
        self.ddim_steps = 100
        if self.opt.debug == "1":
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.vqvae = nn.parallel.DistributedDataParallel(
            self.vqvae,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
        )
        self.img_enc = nn.parallel.DistributedDataParallel(
            self.img_enc,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.img_linear = nn.parallel.DistributedDataParallel(
            self.img_linear,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        self.txt_enc = nn.parallel.DistributedDataParallel(
            self.txt_enc,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, uc_scale=1., opt=None):
        
        df_conf = OmegaConf.load(opt.df_cfg)
        df_model_params = df_conf.model.params
        
        # ref: ddpm.py, line 44 in __init__()
        self.parameterization = "eps"
        self.learn_logvar = False
        
        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule(
            timesteps=df_model_params.timesteps,
            linear_start=df_model_params.linear_start,
            linear_end=df_model_params.linear_end,
        )
        
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        # for cls-free guidance
        self.uc_scale = uc_scale

        # for diffusion
    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()

    def set_input(self, input=None, gen_order=None, max_sample=None):
        
        self.x = input['sdf']
        BS = self.x.shape[0]

        self.img = input['img']
        self.uc_img = torch.zeros_like(self.img).to(self.device)

        self.txt = input['text']
        self.uc_txt = BS * [""]

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.img = self.img[:max_sample]
            self.uc_img = self.uc_img[:max_sample]
            self.txt = self.txt[:max_sample]
            self.uc_txt = self.uc_txt[:max_sample]

        vars_list = ['x', 'img']

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()
        self.img_enc.train()
        self.txt_enc.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.img_enc.eval()
        self.txt_enc.eval()

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    # check: ddpm.py, line 891
    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            # key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            key = 'c_concat' if self.df_module.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        # eps
        out = self.df(x_noisy, t, **cond)

        if isinstance(out, tuple) and not return_ids:
            return out[0]
        else:
            return out

    def get_loss(self, pred, target, loss_type='l2', mean=True):
        if loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_model
    def p_losses(self, x_start, cond, t, noise=None):

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # predict noise (eps) or x0
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # l2
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'loss_total': loss.clone().detach().mean()})

        return x_noisy, target, loss, loss_dict


    def forward(self):

        self.switch_train()

        bs = self.x.shape[0]
        c_img = self.img_enc(self.img) # bs, 64, 2048
        c_txt = self.txt_enc(self.txt) # bs, 77, 1280

        p_img = torch.rand(bs, device=self.device) > 0.5
        p_txt = torch.rand(bs, device=self.device) > 0.5

        c_img = self.img_linear(c_img) * p_img[:, None, None]
        c_txt = c_txt * p_txt[:, None, None]

        c_mm = torch.cat([c_img, c_txt], dim=1)

        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        #    check: ldm.models.autoencoder.py, VQModelInterface's encode(self, x)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach()

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c_mm, t)

        self.loss_df = loss
        self.loss_dict = loss_dict

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
    # def inference(self, data, sample=True, ddim_steps=None, ddim_eta=0., quantize_denoised=True, infer_all=False):
    def inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):

        self.switch_eval()

        if not infer_all:
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        B = self.x.shape[0]

        # get uc/c features from img and txt
        img_uc_feat = self.img_enc(self.uc_img)
        txt_uc_feat = self.txt_enc(self.uc_txt)

        img_uc_feat = self.img_linear(img_uc_feat)
        mm_uc_feat = torch.cat([img_uc_feat, txt_uc_feat], dim=1)

        c_img = self.img_enc(self.img)
        c_txt = self.txt_enc(self.txt)

        c_img = self.img_linear(c_img)
        c_mm = torch.cat([c_img, c_txt], dim=1)
        shape = self.z_shape

        # get noise, denoise, and decode with vqvae
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                        batch_size=B,
                                                        shape=shape,
                                                        conditioning=c_mm,
                                                        verbose=False,
                                                        unconditional_guidance_scale=uc_scale,
                                                        unconditional_conditioning=mm_uc_feat,
                                                        eta=ddim_eta,
                                                        quantize_x0=False)

        self.gen_df = self.vqvae_module.decode_no_quant(samples)
        self.switch_train()

    # def mm_inference(self, data, ddim_steps=None, ddim_eta=0., uc_scale=3.,
    #         txt_scale=1.0, img_scale=1.0, mask_mode='1', mask_x=False,
    #         mm_cls_free=False,
    #     ):
    def mm_inference(self, data, mask_mode=None, ddim_steps=None, ddim_eta=0., uc_scale=None, 
                  txt_scale=1.0, img_scale=1.0, mm_cls_free=False, infer_all=False, max_sample=16):
    
        self.switch_eval()

        if not infer_all:
            # max_sample = 16
            self.set_input(data, max_sample=max_sample)
        else:
            self.set_input(data)

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale
            
            
        # get noise, denoise, and decode with vqvae
        B = self.x.shape[0]
        z = self.vqvae(self.x, forward_no_quant=True, encode_only=True)

        # get mask
        from utils.demo_util import get_shape_mask
        x_mask, z_mask = get_shape_mask(mask_mode)

        # for vis purpose
        self.x_part = self.x.clone()
        self.x_part[~x_mask] = 0.2

        shape = self.z_shape

        # get feat
        img_uc_feat = self.img_enc(self.uc_img)
        img_uc_feat = self.img_linear(img_uc_feat)
        txt_uc_feat = self.txt_enc(self.uc_txt)

        print(f'[mm inference]: t: {txt_scale}, i:{img_scale}')
        c_img = self.img_enc(self.img)
        c_img = self.img_linear(c_img)
        c_txt = self.txt_enc(self.txt)

        # naive inference mode
        if not mm_cls_free:
            
            img_uc_feat *= img_scale
            txt_uc_feat *= txt_scale
            mm_uc_feat = torch.cat([img_uc_feat, txt_uc_feat], dim=1)

            c_img *= img_scale
            c_txt *= txt_scale
            c_mm = torch.cat([c_img, c_txt], dim=1)

            samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                            batch_size=B,
                                                            shape=shape,
                                                            conditioning=c_mm,
                                                            verbose=False,
                                                            x0=z,
                                                            mask=z_mask,
                                                            unconditional_guidance_scale=uc_scale,
                                                            unconditional_conditioning=mm_uc_feat,
                                                            eta=ddim_eta,)
        else:
            c_mm = {
                'c_img': c_img,
                'c_txt': c_txt,
                'img_w': img_scale,
                'txt_w': txt_scale,
            }
            uc_mm = {
                'uc_img': img_uc_feat,
                'uc_txt': txt_uc_feat,
            }
            samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                batch_size=B,
                                                shape=shape,
                                                conditioning=c_mm,
                                                verbose=False,
                                                x0=z,
                                                mask=z_mask,
                                                unconditional_guidance_scale=uc_scale,
                                                unconditional_conditioning=uc_mm,
                                                eta=ddim_eta,
                                                mm_cls_free=True)

        self.gen_df = self.vqvae_module.decode_no_quant(samples)

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        
        self.switch_eval()
        return ret


    def backward(self):
        
        self.loss = self.loss_df

        self.loss_dict = reduce_loss_dict(self.loss_dict)
        self.loss_total = self.loss_dict['loss_total']
        self.loss_simple = self.loss_dict['loss_simple']
        self.loss_vlb = self.loss_dict['loss_vlb']
        if 'loss_gamma' in self.loss_dict:
            self.loss_gamma = self.loss_dict['loss_gamma']

        self.loss.backward()

    def optimize_parameters(self, total_steps):

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_logs_data(self):
        """ return a dictionary with
            key: graph name
            value: an OrderedDict with the data to plot
        
        """
        raise NotImplementedError
        return ret

    def get_current_errors(self):
        
        ret = OrderedDict([
            ('total', self.loss_total.data),
            ('simple', self.loss_simple.data),
            ('vlb', self.loss_vlb.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret

    def write_text_on_img(self, text, bs=16, img_shape=(3, 256, 256)):
        # write text as img
        b, c, h, w = len(text), 3, 256, 256
        img_text = np.ones((b, h, w, 3)).astype(np.float32) * 255
        # font = cv2.FONT_HERSHEY_PLAIN
        font = cv2.FONT_HERSHEY_COMPLEX
        font_size = 0.5
        n_char_per_line = 25 # new line for text

        y0, dy = 20, 1
        for ix, txt in enumerate(text):
            # newline every "space" chars
            for i in range(0, len(txt), n_char_per_line):
                y = y0 + i * dy
                # new_txt.append(' '.join(words[i:i+space]))
                # txt_i = ' '.join(txt[i:i+space])
                txt_i = txt[i:i+n_char_per_line]
                cv2.putText(img_text[ix], txt_i, (10, y), font, font_size, (0., 0., 0.), 2)

        return img_text/255.

    def get_current_visuals(self):

        with torch.no_grad():

            self.img = self.img
            self.txt = self.txt
            self.img_gt = render_sdf(self.renderer, self.x)
            self.img_gen_df = render_sdf(self.renderer, self.gen_df)

        b, c, h, w = self.img_gt.shape
        img_shape = (3, h, w)
        # write text as img
        self.img_text = self.write_text_on_img(self.txt, bs=b, img_shape=img_shape)
        self.img_text = rearrange(torch.from_numpy(self.img_text), 'b h w c -> b c h w')

        vis_tensor_names = [
            'img',
            'img_gt',
            'img_gen_df',
            'img_text',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
        return OrderedDict(visuals)

    def save(self, label, global_step, save_opt=False):
        
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'img_enc': self.img_enc_module.state_dict(),
            'img_linear': self.img_linear_module.state_dict(),
            'txt_enc': self.txt_enc_module.state_dict(),
            'df': self.df_module.state_dict(),
            # 'opt': self.optimizer.state_dict(),
        }
        
        if save_opt:
            state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        torch.save(state_dict, save_path)


    def load_ckpt(self, ckpt, load_opt=False):

        # need this line or you will never be able to run inference code...
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.img_enc.load_state_dict(state_dict['img_enc'])
        self.img_linear.load_state_dict(state_dict['img_linear'])
        self.txt_enc.load_state_dict(state_dict['txt_enc'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


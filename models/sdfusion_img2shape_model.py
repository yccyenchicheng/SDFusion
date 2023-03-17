# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py
import os
from collections import OrderedDict
from functools import partial
from inspect import isfunction

import numpy as np
import einops
import mcubes
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
from models.networks.clip_networks.network import CLIPImageEncoder
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

class SDFusionImage2ShapeModel(BaseModel):
    def name(self):
        return 'SDFusionImage2ShapeModel'

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
        clip_param = df_conf.clip.params
        self.cond_model = CLIPImageEncoder(model=clip_param.model)
        self.cond_model.to(self.device)
        for param in self.cond_model.parameters():
            param.requires_grad = True # fine-tune clip
        ######## END: Define Networks ########
        
        # params
        trainable_models = [self.df, self.cond_model]
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
            self.cond_model_module = self.cond_model.module
        else:
            self.df_module = self.df
            self.vqvae_module = self.vqvae
            self.cond_model_module = self.cond_model

        # for debugging purpose
        self.ddim_steps = 100
        if self.opt.debug == "1":
            self.ddim_steps = 20
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

        self.cond_model = nn.parallel.DistributedDataParallel(
            self.cond_model,
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
        self.img = input['img']
        self.uc_img = torch.zeros_like(self.img).to(self.device)

        if max_sample is not None:
            self.x = self.x[:max_sample]
            self.img = self.img[:max_sample]
            self.uc_img = self.uc_img[:max_sample]

        vars_list = ['x', 'img']

        self.tocuda(var_names=vars_list)

    def switch_train(self):
        self.df.train()
        self.cond_model.train()

    def switch_eval(self):
        self.df.eval()
        self.vqvae.eval()
        self.cond_model.eval()

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
        # prefix = 'train' if self.training else 'val'

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

        c_img = self.cond_model(self.img).float()
        
        # 1. encode to latent
        #    encoder, quant_conv, but do not quantize
        #    check: ldm.models.autoencoder.py, VQModelInterface's encode(self, x)
        with torch.no_grad():
            z = self.vqvae(self.x, forward_no_quant=True, encode_only=True).detach()

        # 2. do diffusion's forward
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        z_noisy, target, loss, loss_dict = self.p_losses(z, c_img, t)

        self.loss_df = loss
        self.loss_dict = loss_dict

    # check: ddpm.py, log_images(). line 1317~1327
    @torch.no_grad()
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

        # get noise, denoise, and decode with vqvae
        uc = self.cond_model(self.uc_img).float() # img shape
        c_img = self.cond_model(self.img).float()
        B = c_img.shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                        batch_size=B,
                                                        shape=shape,
                                                        conditioning=c_img,
                                                        verbose=False,
                                                        unconditional_guidance_scale=uc_scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        quantize_x0=False)


        self.gen_df = self.vqvae_module.decode_no_quant(samples)

        self.switch_train()

    @torch.no_grad()
    def img2shape(self, image, mask, ddim_steps=None, ddim_eta=0., uc_scale=None,
                  infer_all=False, max_sample=16):
        #######################
        ### preprocess data ###
        from utils.demo_util import preprocess_image
        import torchvision.transforms as transforms
        
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.Resize((256, 256)),
        ])
        
        _, img = preprocess_image(image, mask)
        img = transforms(img)
        self.img = img.unsqueeze(0).to(self.device)
        self.uc_img = torch.zeros_like(self.img).to(self.device)
        #######################

        # real inference
        self.switch_eval()

        if ddim_steps is None:
            ddim_steps = self.ddim_steps

        if uc_scale is None:
            uc_scale = self.uc_scale

        # get noise, denoise, and decode with vqvae
        uc = self.cond_model(self.uc_img).float() # img shape
        c_img = self.cond_model(self.img).float()
        B = c_img.shape[0]
        shape = self.z_shape
        samples, intermediates = self.ddim_sampler.sample(S=ddim_steps,
                                                        batch_size=B,
                                                        shape=shape,
                                                        conditioning=c_img,
                                                        verbose=False,
                                                        unconditional_guidance_scale=uc_scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        quantize_x0=False)


        self.gen_df = self.vqvae_module.decode_no_quant(samples)

        return self.gen_df


    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.switch_eval()
        
        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])

        self.switch_train()
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
        
        self.set_requires_grad([self.df], requires_grad=True)
        self.set_requires_grad([self.cond_model], requires_grad=True)
        
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

    def get_current_visuals(self):

        with torch.no_grad():
            self.img = self.img # input image
            self.img_gt = render_sdf(self.renderer, self.x) # rendered gt sdf
            self.img_gen_df = render_sdf(self.renderer, self.gen_df) # rendered generated sdf
            
        vis_tensor_names = [
            'img',
            'img_gt',
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)
                            
        return OrderedDict(visuals)

    def save(self, label, global_step, save_opt=False):
        
        state_dict = {
            'vqvae': self.vqvae_module.state_dict(),
            'cond_model': self.cond_model_module.state_dict(),
            'df': self.df_module.state_dict(),
            'global_step': global_step,
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
        self.cond_model.load_state_dict(state_dict['cond_model'])
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))


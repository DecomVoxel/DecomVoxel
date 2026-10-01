import os
import os.path as osp

from copy import deepcopy
from typing import Union, Dict, List
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageFilter
import cv2
import numpy as np

import torch
import torch.utils.data
import torch.utils
import torch.nn as nn
from torchvision import transforms as TF
from diffusers import DDIMScheduler, AutoencoderKL

def mask_dilate(mask, kernel_size):
        if type(mask) != np.ndarray:
            mask = np.array(mask)

        if kernel_size!=0:
            mask = cv2.dilate(mask,
                            np.ones((kernel_size, kernel_size), np.uint8),
                            iterations=1)
        return mask

def mask_morphologyEx(mask, kernel_size):
        if type(mask) != np.ndarray:
            mask = np.array(mask)

        if kernel_size!=0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((kernel_size, kernel_size), np.uint8),
                            iterations=1)
        return mask


def get_timesteps(scheduler, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(
            int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order:]

        return timesteps, num_inference_steps - t_start

def predict_noise(diff_model, noisy_latents, resized_masks, masked_latents, timesteps, input_ids, guidance_scale=1.0):
    CFG_GUIDANCE = guidance_scale != 1 

    if CFG_GUIDANCE:
        noisy_latents = torch.cat([noisy_latents] * 2)
        resized_masks = torch.cat([resized_masks] * 2)
        masked_latents = torch.cat([masked_latents] * 2)

        assert input_ids.shape[0] % 2 == 0

    latent_model_input = torch.cat([
        noisy_latents, resized_masks, masked_latents], dim=1)

    # Predict the noise residual
    noise_pred = diff_model(
        latent_model_input,
        timesteps=timesteps,
        input_ids=input_ids
    ).sample

    if CFG_GUIDANCE:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond)

    return noise_pred  

class PixelHacker_Pipeline:
    def __init__(self,
        model: nn.Module,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        device='cuda',
        dtype=torch.float16,
    ):
        # prepare model
        self.vae = vae
        self.vae.to(device=device, dtype=dtype)
        self.vae.eval()
        self.model = model
        self.model.to(device=device, dtype=dtype)
        self.model.eval()

        # scheduler
        self.noise_scheduler = scheduler

        self.vae_ds_ratio = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.device = device
        self.dtype = dtype

        half_id_num = self.model.num_embeddings // 2
        id_num = self.model.num_embeddings
        print('[Info]: model.num_embeddings: {}, input_half_id_num: {}, input_id_num: {}'.format(self.model.num_embeddings, half_id_num, id_num))

        fg = torch.tensor([list(range(half_id_num))], dtype=torch.int64, device=self.device, requires_grad=False)
        bg = torch.tensor([list(range(half_id_num, id_num))], dtype=torch.int64, device=self.device, requires_grad=False)
        self.input_ids = torch.cat([bg, fg]).to(device=self.device)

        print('[Info]: Load pipeline successed.')

    def _preprocess(self,
                    input_image,
                    input_mask,
                    image_size,
                    mask_dilate_kernel_size=0,
                    mask_preprocess_type='dilate'):
        ###### step 1######
        threshold = 255/2
        input_mask = input_mask.point(lambda x: 0 if x < threshold else 255, 'L')

        input_image_copy = deepcopy(input_image)
        mask_image_copy = deepcopy(input_mask)
        size1, size2 = input_image.size

        w, h = input_image.size
        if w < h:
            scale = image_size / w
            w_t, h_t = image_size, int(h*scale)
        else:
            scale = image_size / h
            w_t, h_t = int(w*scale), image_size
        w_t, h_t = w_t // 64 * 64, h_t // 64 * 64

        input_image = input_image.resize((w_t, h_t))
        input_mask = input_mask.resize((w_t, h_t))

        input_mask = self.mask_preprocess(input_mask, mask_dilate_kernel_size, kind=mask_preprocess_type)

        image, mask, masked_image = self._denoise_preprocess(
            input_image, input_mask)

        _info = SimpleNamespace(
            input_image_copy = input_image_copy,
            mask_image_copy = mask_image_copy,
            size1 = size1, 
            size2 = size2
        )

        return image, mask, masked_image, _info
    
    @staticmethod
    def mask_preprocess(mask_image, mask_dilate_kernel_size, kind='dilate'):
        assert kind in ('dilate', 'morphologyEx')
        if kind == 'dilate':
            mask = mask_dilate(mask_image, mask_dilate_kernel_size)
        elif kind == 'morphologyEx':
            mask = mask_morphologyEx(mask_image, mask_dilate_kernel_size)
        return Image.fromarray(mask)

    def prepare_mask_and_masked_image(self, image, mask):
        image = image.float()
        mask = torch.where(mask >= 0.5, 1, 0).unsqueeze(0).to(device=self.device)

        masked_image = image * (1 - mask)
        masked_image = masked_image.to(dtype=self.dtype, device=self.device)

        return mask.to(dtype=torch.uint8), masked_image

    def _denoise_preprocess(self,image_migan,input_mask):
        # ####### step 3 ############
        image = np.asarray(image_migan) / 255. *2 - 1
        image = torch.tensor(image).permute(2, 0, 1)
        image = image.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        mask = np.asarray(input_mask) / 255.
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0).to(dtype=self.dtype, device=self.device)

        mask, masked_image = self.prepare_mask_and_masked_image(image, mask)

        return image, mask, masked_image

    @staticmethod
    def _post_process(images, _info, paste=False, compensate=False):
        size1 = _info.size1
        size2 = _info.size2
        input_image_copy = _info.input_image_copy
        mask_image_copy = _info.mask_image_copy

        ######## step 4#######
        image = images[0]
        image = image.permute(1, 2, 0)
        image = Image.fromarray((torch.clamp(image.float(), 0, 1)*255).cpu().numpy().astype(np.uint8))

        result_resize = image.resize((size1, size2),resample=Image.Resampling.LANCZOS)
        if paste and not compensate:
            m_img = mask_image_copy.convert('RGB').filter(
                ImageFilter.GaussianBlur(radius=3))
            m_img = np.asarray(m_img) / 255.0
            img_np = np.asarray(input_image_copy.convert('RGB')) / 255.0
            ours_np = np.asarray(result_resize) / 255.0
            ours_np = ours_np * m_img + (1 - m_img) * img_np
            out_arr = np.uint8(ours_np * 255)
            out_sample = Image.fromarray(out_arr)
        elif paste and compensate:
            m_img = mask_image_copy.resize((size1, size2), Image.NEAREST)
            _img = input_image_copy.resize((size1, size2), Image.LANCZOS)
            out_sample = paste_compensate(mask_image_copy, input_image_copy, result_resize, fac=1.1)
        else:
            out_sample = result_resize

        return out_sample

    def _denoise_steps(self,image,mask,masked_image,
                       num_steps=20,
                       strength=0.999,
                       noise_offset=None,
                       guidance_scale=4.5,
                       mute=True):
        # set_timesteps
        self.noise_scheduler.set_timesteps(
            num_inference_steps=num_steps, device=self.device)
        timesteps, num_inference_steps = get_timesteps(
            self.noise_scheduler,
            num_inference_steps=num_steps,
            strength=strength,
            device=self.device)
        latent_timestep = timesteps[:1] #.repeat(2 * 1)

        with torch.no_grad():
            # latent
            latents = self.vae.encode(image).latent_dist.sample()
            latents *= self.vae.config.scaling_factor
            masked_latents = self.vae.encode(masked_image).latent_dist.sample() 
            masked_latents *= self.vae.config.scaling_factor

            # resize mask
            h, w = mask.shape[-2:]
            size = (h // self.vae_ds_ratio, w // self.vae_ds_ratio)
            resized_masks = torch.nn.functional.interpolate(mask, size=size).to(device=self.device, dtype=self.dtype)

            # add noise
            noise = torch.randn_like(latents)
            if noise_offset:
                noise += noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, latent_timestep) if strength < 1 else noise

            for _, t in tqdm(enumerate(timesteps),disable=mute):
                t = t.to(device=self.device).unsqueeze(0)

                bg, fg = self.input_ids.chunk(2)
                if guidance_scale != 1:
                    input_ids = torch.cat([
                        bg.repeat(latents.shape[0],1),
                        fg.repeat(latents.shape[0],1)
                    ])
                else:
                    input_ids = torch.cat([fg.repeat(latents.shape[0],1)])

                noisy_latents = self.noise_scheduler.scale_model_input(noisy_latents, t)

                # Predict the noise residual
                noise_pred = predict_noise(
                    self.model,
                    noisy_latents,
                    resized_masks=resized_masks,
                    masked_latents=masked_latents,
                    timesteps=t,
                    input_ids=input_ids,
                    guidance_scale=guidance_scale
                )

                noisy_latents = self.noise_scheduler.step(noise_pred, t, noisy_latents, return_dict=False)[0]

            images = self.vae.decode((noisy_latents/self.vae.config.scaling_factor).to(self.dtype)).sample

            images = (images+1)/2
        return images

    def __call__(
            self,
            input_image_list: List[Image.Image],
            input_mask_list: List[Image.Image],
            image_size=512,
            mask_dilate_kernel_size=0,
            mask_preprocess_type='dilate',
            strength=0.999,
            num_steps=20,
            guidance_scale=4.5,
            retry=0,
            paste=False,
            compensate=False,
            noise_offset = None, #0.0357
            mute=True):
        if retry == 0:
            seed = 0
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        if not isinstance(input_image_list, list):
            input_image_list = [input_image_list]
        if not isinstance(input_mask_list, list):
            input_mask_list = [input_mask_list]

        batch_image, batch_mask, batch_masked_image = [], [], []
        _info_list = []
        for input_image, input_mask in zip(input_image_list, input_mask_list):
            image, mask, masked_image, _info = self._preprocess(
            input_image, input_mask, image_size, 
            mask_dilate_kernel_size=mask_dilate_kernel_size, 
            mask_preprocess_type=mask_preprocess_type,)
            batch_image.append(image)
            batch_mask.append(mask)
            batch_masked_image.append(masked_image)
            _info_list.append(_info)

        batch_image = torch.cat(batch_image)
        batch_mask = torch.cat(batch_mask)
        batch_masked_image = torch.cat(batch_masked_image)

        images = self._denoise_steps(
            batch_image, batch_mask, batch_masked_image,
            num_steps=num_steps,
            strength=strength,
            noise_offset=noise_offset,
            guidance_scale=guidance_scale,
            mute=mute
        )

        out_sample_list = []
        for image, _info in zip(images.split(1), _info_list):
            out_sample = self._post_process(
                image, _info, paste=paste, compensate=compensate
            )
            out_sample_list.append(out_sample)
        return out_sample_list
    


def get_masked_stat(img:np.ndarray, mask=None, per_channel = True):
    ax = (0,1) if per_channel else None
    ret = dict(mean=img.mean(ax), std=img.std(ax))

    if mask is not None:
        mask = np.stack([mask]*3, axis=-1)
        unmask = 1 - mask
        neighbor_unmask = np.stack(
            [get_bbox_unmask(mask[...,0])]*3, axis=-1)

        masked_pixels = mask.sum(ax)
        masked_mean = (img*mask).sum(ax)/masked_pixels
        # breakpoint()
        masked_squared_diffs = ((img - masked_mean) ** 2) * mask
        masked_var = masked_squared_diffs.sum(ax) / masked_pixels
        masked_std = np.sqrt(masked_var)

        unmasked_pixels = unmask.sum(ax)
        unmasked_mean = (img*unmask).sum(ax)/unmasked_pixels
        unmasked_squared_diffs = ((img - unmasked_mean) ** 2) * unmask
        unmasked_var = unmasked_squared_diffs.sum(ax) / unmasked_pixels
        unmasked_std = np.sqrt(unmasked_var)

        nbr_unmasked_pixels = neighbor_unmask.sum(ax)
        nbr_unmasked_mean = (img*neighbor_unmask).sum(ax)/nbr_unmasked_pixels
        nbr_unmasked_squared_diffs = ((img - nbr_unmasked_mean) ** 2) * neighbor_unmask
        nbr_unmasked_var = nbr_unmasked_squared_diffs.sum(ax) / nbr_unmasked_pixels
        nbr_unmasked_std = np.sqrt(nbr_unmasked_var)


        # ret['count']=masked_pixels
        ret['masked_mean']=masked_mean
        ret['masked_std']=masked_std
        ret['masked_var']=masked_var
        # ret['masked_sq_err']=masked_squared_diffs
        
        ret['unmasked_mean']=unmasked_mean
        ret['unmasked_std']=unmasked_std
        ret['unmasked_var']=unmasked_var

        ret['nbr_unmasked_mean']=nbr_unmasked_mean
        ret['nbr_unmasked_std']=nbr_unmasked_std
        ret['nbr_unmasked_var']=nbr_unmasked_var

    return ret

def get_unmasked_stat(img:np.ndarray, mask=None, per_channel = True):
    ''' a minimal ver. of get_masked_stat '''
    ax = (0,1) if per_channel else None
    ret = dict(mean=img.mean(ax), std=img.std(ax))

    if mask is not None:
        mask = np.stack([mask]*3, axis=-1)
        neighbor_unmask = np.stack(
            [get_bbox_unmask(mask[...,0])]*3, axis=-1)

        nbr_unmasked_pixels = neighbor_unmask.sum(ax)
        nbr_unmasked_mean = (img*neighbor_unmask).sum(ax)/nbr_unmasked_pixels

        ret['nbr_unmasked_mean']=nbr_unmasked_mean

    return ret



def add_text(image:Image, text, position=(0,0), fontsize=16):
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(fontsize)
    draw.multiline_text(position, text, fill='violet', font=font)

def add_stat(img:Image, st, position=(0,240), fontsize=32):
    format_ = lambda arr: ', '.join(map(lambda v: f"{v:.2f}", arr))
    total_means = format_(st['mean'])
    total_stds = format_(st['std'])
    masked_means = format_(st['masked_mean'])
    masked_stds = format_(st['masked_std'])
    unmasked_means = format_(st['unmasked_mean'])
    unmasked_stds = format_(st['unmasked_std'])

    msg = [f"total_mean:{total_means}",
           f"total_std:{total_stds}",
           "",
           f"masked_mean:{masked_means}",
           f"masked_std:{masked_stds}",
           "",
           f"unmasked_mean:{unmasked_means}",
           f"unmasked_std:{unmasked_stds}"]
    
    if 'nbr_unmasked_mean' in st.keys():
        nbr_unmasked_means = format_(st['nbr_unmasked_mean'])
        nbr_unmasked_stds = format_(st['nbr_unmasked_std'])
        msg.extend(["",
            f"nbr_unmasked_mean:{nbr_unmasked_means}",
            f"nbr_unmasked_std:{nbr_unmasked_stds}"])

    add_text(img, '\n'.join(msg), position, fontsize)



def get_normalize(ori:np.ndarray,ipt:np.ndarray,mask:np.ndarray,stat:dict) -> Image:
    mask = np.stack([mask]*3, axis=-1)

    from PIL import ImageFilter
    m_img = Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    blur_mask = np.asarray(m_img) / 255.0
  
    ours_np_ = (ipt.astype(np.float32) - stat['ipt']['masked_mean'])/stat['ipt']['masked_std']
    ours_np_ = ours_np_ * stat['ori']['unmasked_std'] + stat['ori']['unmasked_mean']
    ours_np = ours_np_ / 255.  # ours_np = ipt / 255.0
    
    img_np = ori / 255.0

    ours_np = ours_np * blur_mask + (1 - blur_mask) * img_np
    image_inpaint_compensate = np.uint8(ours_np * 255)
    return Image.fromarray(image_inpaint_compensate.astype(np.uint8))



def get_compensation(ori:np.ndarray,ipt:np.ndarray,mask:np.ndarray,delta_mean:float, fac=1.0) -> Image:
    mask = np.stack([mask]*3, axis=-1)

    from PIL import ImageFilter
    m_img = Image.fromarray((mask*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=3))
    blur_mask = np.asarray(m_img) / 255.0  # blur_mask = mask #

    ori_ = ori/255.0
    ipt_ = ipt/255.0
    cps = (ipt_ + fac * delta_mean/255.0) * blur_mask + (1 - blur_mask) * ori_
    
    image_inpaint_compensate = np.uint8(cps.clip(0, 1.) * 255)
    return Image.fromarray(image_inpaint_compensate)


def get_bbox_unmask(mask:np.ndarray, off=10)-> np.ndarray:
    rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
    
    min_r, max_r = np.argmax(rows), len(rows) - np.argmax(rows[::-1]) - 1
    min_c, max_c = np.argmax(cols), len(cols) - np.argmax(cols[::-1]) - 1

    # Adjust the box size
    h, w = max_r - min_r + 1, max_c - min_c + 1
    new_h, new_w = int(h + off), int(w + off) # new_h, new_w = int(h * fac), int(w * fac)
    
    # Cal new center
    c_r, c_c = (min_r + max_r) // 2, (min_c + max_c) // 2
    
    # Cal new corners
    new_min_r = max(0, c_r - new_h // 2)
    new_max_r = min(mask.shape[0] - 1, c_r + new_h // 2)
    new_min_c = max(0, c_c - new_w // 2)
    new_max_c = min(mask.shape[1] - 1, c_c + new_w // 2)

    new_mask = np.zeros_like(mask, dtype=np.bool_)
    new_mask[new_min_r:new_max_r + 1, new_min_c:new_max_c + 1] = True
    new_mask[np.where(mask == True)] = False

    return new_mask


def paste_compensate(mask:Image, image:Image, result:Image, fac=1.0):
    np_img = np.asarray(image)
    np_msk = np.asarray(mask) > 128
    np_ipt = np.asarray(result)
    ori_stat = get_unmasked_stat(np_img, np_msk)
    ipt_stat = get_unmasked_stat(np_ipt, np_msk)

    delta_mean = ori_stat['nbr_unmasked_mean'] - ipt_stat['nbr_unmasked_mean']

    return get_compensation(np_img, np_ipt, np_msk, delta_mean, fac=fac)

"""SAMPLING ONLY."""

import torch
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm
from functools import partial
from torch.nn.functional import mse_loss, l1_loss
from torchvision import transforms
from torchvision.utils import save_image
import torchvision

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor

from torchvision.transforms.functional import gaussian_blur 
from PIL import Image

# this is for target guided
class DDIMSampler(object):
    def __init__(self, model, schedule="linear", tgt_scale=0.01, guide_gaze=False, guide_fq=False, **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.tgt_scale = tgt_scale
        self.list_GazeLoss = []
        self.list_SegmentLoss = []
        self.list_totaloss = []
        self.guide_gaze = guide_gaze
        self.guide_fq = guide_fq


    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               timesteps=None,
               dict_condfn=None,
               decode=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    timesteps=timesteps,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dict_condfn = dict_condfn,
                                                    decode = decode
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dict_condfn=None, decode=None):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Target Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            outs = self.p_sample_ddim(img, cond, ts, x0=x0, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dict_condfn=dict_condfn, decode=decode)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, x0=None, dict_condfn=None, decode=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)

        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        
        e_t = e_t - self.tgt_scale * a_t.sqrt() * (pred_x0 - x0)

        
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        # Check condition model and add guide
        if dict_condfn is not None:
            e_t = e_t -  self.tgt_scale * a_t.sqrt() * (self.calculate_gradient(dict_condfn, x_prev, t, x0, decode))
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
            if noise_dropout > 0.:
                noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

        return x_prev, pred_x0
    
    def complex_mse_loss(self, output, target):
        return (0.5*(abs(output - target))**2).mean(dtype=torch.complex64)

    def calculate_gradient(self, dict_condfn, x_prev, t, x0, decode):
        with torch.enable_grad():
            loss = torch.tensor(0)
            x_prev = x_prev.detach().requires_grad_(True)

            predicted_image = decode(x_prev)
            predicted_image = (predicted_image+1.0)/2.0

            target_image = decode(x0)
            target_image = (target_image+1.0)/2.0

            # loss fft
            if self.guide_fq: 
                src_mask_f  = predicted_image
                img_gray_s = torchvision.transforms.functional.rgb_to_grayscale(src_mask_f)
                f_s = torch.fft.fft2(img_gray_s)

                targ_mask_f = target_image 
                img_gray_t = torchvision.transforms.functional.rgb_to_grayscale(targ_mask_f)
                f_t =  torch.fft.fft2(img_gray_t)

                loss = loss + self.complex_mse_loss(f_s, f_t)*5

            # loss Gaze
            if self.guide_gaze:
                gaze_loss = self.calculte_loss_gaze(dict_condfn['gaze'], target_image=target_image, predicted_image=predicted_image, t=t, fa = dict_condfn['fa'])
                self.list_GazeLoss.append(gaze_loss.sum().detach().cpu().numpy())
                loss = loss + gaze_loss.sum()

            # Check valid loss and calculate gradient
            if loss.detach().cpu().numpy() != 0:
                try:
                    grad_result = torch.autograd.grad(loss, x_prev)[0]
                except RuntimeError as e:
                    print(f"Error for y: {e}")
                if grad_result is not None:
                    gradient = grad_result
                else:
                    gradient = 0
            else:
                gradient = 0

        return gradient
    
    def decode_img(self, samples, decode):
        x_samples = decode(samples.to(samples.device))

        gen_imgs = torch.clamp((x_samples+1.0)/2.0, min=0.0, max=1.0)
        gen_imgs = (gen_imgs * 255).to(torch.uint8)
        # gen_imgs = rearrange(gen_imgs, 'b c h w -> b h w c')
        return gen_imgs
    
    def get_eye_coords(self, fa, image):
        image = image.squeeze(0)
        image = image * 128 + 128
        image = image.to(torch.uint8)
        image = image.permute(1, 2, 0) # h w c
        image = image.cpu().numpy()

        try:
            preds = fa.get_landmarks(image)[0]
        except:
            return [None] * 8

        x, y = 5, 9
        left_eye_left = preds[36]
        left_eye_right = preds[39]
        eye_y_average = (left_eye_left[1] + left_eye_right[1]) // 2
        left_eye = [int(left_eye_left[0]) - x, int(eye_y_average - y), int(left_eye_right[0]) + x, int(eye_y_average + y)]
        right_eye_left = preds[42]
        right_eye_right = preds[45]
        eye_y_average = (right_eye_left[1] + right_eye_right[1]) // 2
        right_eye = [int(right_eye_left[0]) - x, int(eye_y_average - y), int(right_eye_right[0]) + x, int(eye_y_average + y)]
        return [*left_eye, *right_eye]

    def calculte_loss_gaze(self, model, target_image, predicted_image, t, fa):
        # Gaze loss
        gaze_loss = torch.tensor(0)

        # choice time to add gaze guide
        guide_at_t = 150 
        if t < guide_at_t:# and t > 5:
            src_eye = predicted_image
            # src_eye = src_eye * 0.5 + 0.5
            # save_image(predicted_image,f'data/result_step/{int(t.detach().cpu().numpy())}_predicted_image.png')
            
            targ_eye = target_image
            # targ_eye = targ_eye * 0.5 + 0.5
            # save_image(target_image,'target_image.png')
            llx, lly, lrx, lry, rlx, rly, rrx, rry = self.get_eye_coords(fa, targ_eye)

            if llx is not None:
                targ_left_eye   = targ_eye[:, :, lly:lry, llx:lrx]
                src_left_eye    = src_eye[:, :, lly:lry, llx:lrx]
                targ_right_eye  = targ_eye[:, :, rly:rry, rlx:rrx]
                src_right_eye   = src_eye[:, :, rly:rry, rlx:rrx]
                targ_left_eye   = torch.mean(targ_left_eye.to(torch.float), dim=1, keepdim=True)
                src_left_eye    = torch.mean(src_left_eye.to(torch.float), dim=1, keepdim=True)
                targ_right_eye  = torch.mean(targ_right_eye.to(torch.float), dim=1, keepdim=True)
                src_right_eye   = torch.mean(src_right_eye.to(torch.float), dim=1, keepdim=True)
                targ_left_gaze  = model(targ_left_eye.squeeze(0))
                src_left_gaze   = model(src_left_eye.squeeze(0))
                targ_right_gaze = model(targ_right_eye.squeeze(0))
                src_right_gaze  = model(src_right_eye.squeeze(0))
                left_gaze_loss  = l1_loss(targ_left_gaze, src_left_gaze)
                right_gaze_loss = l1_loss(targ_right_gaze, src_right_gaze)
                gaze_loss = (left_gaze_loss + right_gaze_loss) * 200
            else:
                print('no eye detected')

        return gaze_loss
    
    def calculte_loss_segmentation(self, model, target_image, predicted_image, spNorm=None):
        with torch.enable_grad():
            # Segmentation loss
            src_mask  = (predicted_image + 1) / 2
            src_mask  = transforms.Resize((512,512))(src_mask)
            src_mask  = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(src_mask)
            targ_mask = (target_image + 1) / 2
            targ_mask  = transforms.Resize((512,512))(targ_mask)
            targ_mask = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(targ_mask)

            src_seg  = model(spNorm(src_mask))[0]
            src_seg = transforms.Resize((256, 256))(src_seg)
            targ_seg = model(spNorm(targ_mask))[0]
            targ_seg = transforms.Resize((256, 256))(targ_seg)

            seg_loss = torch.tensor(0).to(target_image.device).float()

            # Attributes = [0, 'background', 1 'skin', 2 'r_brow', 3 'l_brow', 4 'r_eye', 5 'l_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r', 10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
            ids = [1, 2, 3, 4, 5, 10, 11, 12, 13]

            for id in ids:
                seg_loss += l1_loss(src_seg[0,id,:,:], targ_seg[0,id,:,:])
                # seg_loss += mse_loss(src_seg[0,id,:,:], targ_seg[0,id,:,:])

            seg_loss = seg_loss * 10
        return seg_loss 
    
    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec
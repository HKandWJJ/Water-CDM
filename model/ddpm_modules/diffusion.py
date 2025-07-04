import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm



def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# def extract(a, t, x_shape):
#     b, *_ = t.shape
#     out = a.gather(-1, t)
#     return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.conditional = conditional
        self.loss_type = loss_type
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)
        self.eta = 0
        self.sample_proc = 'ddim'
    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss().to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss().to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        ddim_sigma = (self.eta * ((1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas_cumprod / alphas_cumprod_prev)) ** 0.5)
        self.ddim_sigma = to_torch(ddim_sigma)
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # self.register_buffer('ddim_sigma',
        #                      to_torch(ddim_sigma))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, style=None):
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_ddim(self, x, t, clip_denoised: bool, condition_x=None, style=None):
        if condition_x is not None:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style))
            x_recon = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)
        else:
            # x_recon = self.predict_start_from_noise(
            #     x, t=t, noise=self.denoise_fn(x, t))
            x_recon = self.denoise_fn(x, t)
        # if clip_denoised:
        #     x_recon.clamp_(-1., 1.)

        alpha = extract(self.alphas_cumprod, t, x_recon.shape)
        alpha_prev = extract(self.alphas_cumprod_prev, t, x_recon.shape)
        sigma = extract(self.ddim_sigma, t, x_recon.shape)
        sqrt_one_minus_alphas = extract(self.sqrt_one_minus_alphas_cumprod, t, x_recon.shape)
        pred_x0 = (x - sqrt_one_minus_alphas * x_recon) / (alpha ** 0.5)

        dir_xt = torch.sqrt(1. - alpha_prev - sigma ** 2) * x_recon
        noise = torch.randn(x.shape, device=x.device)
        x_prev = (alpha_prev ** 0.5) * pred_x0 + dir_xt + sigma * noise
        return x_prev, pred_x0


    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b,
                                                      *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample2(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)
        logvar = extract(
            self.posterior_log_variance_clipped, t, x.shape)
        weight = bt / torch.sqrt(1 - at)
        et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t, style)
        # if clip_denoised:
        #     et.clamp_(-1., 1.)
        mean = 1 / torch.sqrt(1.0 - bt) * (x - weight * et)
        noise = torch.randn_like(x)
        mask = 1 - (t == 0).float()
        mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))
        xt_next = mean + mask * torch.exp(0.5 * logvar) * noise
        xt_next = xt_next.float()
        return xt_next

    @torch.no_grad()
    def p_sample_ddim(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        x_prev, pred_x0 = self.p_mean_variance_ddim(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, style=style)
        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        return x_prev

    def p_sample_ddim2(self, x, t, t_next, clip_denoised=True, repeat_noise=False, condition_x=None, style=None):
        b, *_, device = *x.shape, x.device
        bt = extract(self.betas, t, x.shape)
        at = extract((1.0 - self.betas).cumprod(dim=0), t, x.shape)

        if condition_x is not None:
            et = self.denoise_fn(torch.cat([condition_x, x], dim=1), t)
        else:
            et = self.denoise_fn(x, t)


        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        # x0_air_t = (x_air - et_air * (1 - at).sqrt()) / at.sqrt()
        if t_next == None:
            at_next = torch.ones_like(at)
        else:
            at_next = extract((1.0 - self.betas).cumprod(dim=0), t_next, x.shape)
        if self.eta == 0:
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
            # xt_air_next = at_next.sqrt() * x0_air_t + (1 - at_next).sqrt() * et_air
        elif at > (at_next):
            print('Inversion process is only possible with eta = 0')
            raise ValueError
        else:
            c1 = self.eta * ((1 - at / (at_next)) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c2 * et + c1 * torch.randn_like(x0_t)
            # xt_air_next = at_next.sqrt() * x0_air_t + c2 * et_air + c1 * torch.randn_like(x0_t)

        # noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0

        return xt_next

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False, cand=None):
        device = self.betas.device
        sample_inter = 10
        g_gpu = torch.Generator(device=device).manual_seed(44444)
        if not self.conditional:
            x = x_in['SR']
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device, generator=g_gpu)
            ret_img = img
            if cand is not None:
                time_steps = np.array(cand)
            else:
                num_timesteps_ddim = np.array([0, 245, 521, 1052, 1143, 1286, 1475, 1587, 1765, 1859])  # searching
                time_steps = np.flip(num_timesteps_ddim)
            for j, i in enumerate(tqdm(time_steps, desc='sampling loop time step', total=len(time_steps))):
                # print('i = ', i)
                t = torch.full((b,), i, device=device, dtype=torch.long)
                if j == len(time_steps) - 1:
                    t_next = None
                else:
                    t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
                img = self.p_sample_ddim2(img, t, t_next, style=x_in['style'])
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
            return img
        else:
            x = x_in['SR']
            shape = x.shape
            b = shape[0]
            img = torch.randn(shape, device=device, generator=g_gpu)
            ret_img = x

            if self.sample_proc == 'ddpm':
                for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                    # print('i = ', i)
                    img = self.p_sample(img, torch.full(
                        (b,), i, device=device, dtype=torch.long), condition_x=x)
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
            else:
                if cand is not None:
                    time_steps = np.array(cand)
                    # print(time_steps)
                else:
                    # time_steps = np.array([1898, 1640, 1539, 1491, 1370, 1136, 972, 858, 680, 340])
                    time_steps = np.array([1900, 1640, 1540, 1490, 1370, 1140, 970, 860, 680, 500, 400, 340])
                    # time_steps = np.asarray(list(range(0, 1000, int(1000/4))) + list(range(1000, 2000, int(1000/6))))
                    # time_steps = np.flip(time_steps[:-1])
                for j, i in enumerate(time_steps):
                    # print('i = ', i)
                    t = torch.full((b,), i, device=device, dtype=torch.long)
                    if j == len(time_steps) - 1:
                        t_next = None
                    else:
                        t_next = torch.full((b,), time_steps[j + 1], device=device, dtype=torch.long)
                    img = self.p_sample_ddim2(img, t, t_next, condition_x=x, style=x_in['style'])
                    if i % sample_inter == 0:
                        ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False, cand=None):
        return self.p_sample_loop(x_in, continous, cand=cand)


    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full(
                (b,), i, device=device, dtype=torch.long))

        return img

    def q_sample_recover(self, x_noisy, t, predict_noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))
        return (x_noisy - extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_noisy.shape) * predict_noise) / extract(self.sqrt_alphas_cumprod, t, x_noisy.shape)


    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )


    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        condition_x = x_in['SR']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, t)
        else:
            x_recon = self.denoise_fn(
                torch.cat([condition_x, x_noisy], dim=1), t)


        loss = self.loss_func(noise, x_recon)

        return loss

    def forward(self, x, flag, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
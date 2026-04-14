import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# MRI forward and adjoint operators
# ------------------------------------------------------------

def fft2c(x):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

def ifft2c(k):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

def forward_op(x, mask, smaps=None):
    """
    x     : (B, H, W) complex image
    mask  : (B, H, W) binary sampling mask
    smaps : optional sensitivity maps (B, C, H, W)
    """
    if smaps is None:
        k = fft2c(x)
        return mask * k
    else:
        x_coils = x.unsqueeze(1) * smaps
        k = fft2c(x_coils)
        return mask.unsqueeze(1) * k

def adjoint_op(y, mask, smaps=None):
    """
    Adjoint MRI operator.
    """
    if smaps is None:
        return ifft2c(mask * y)
    else:
        x_coils = ifft2c(mask.unsqueeze(1) * y)
        return torch.sum(x_coils * torch.conj(smaps), dim=1)

def data_consistency_projection(x, y, mask, smaps=None):
    """
    Hard projection onto measured k-space samples.
    """
    if smaps is None:
        k = fft2c(x)
        k_proj = mask * y + (1.0 - mask) * k
        return ifft2c(k_proj)
    else:
        x_coils = x.unsqueeze(1) * smaps
        k = fft2c(x_coils)
        k_proj = mask.unsqueeze(1) * y + (1.0 - mask.unsqueeze(1)) * k
        x_proj = ifft2c(k_proj)
        return torch.sum(x_proj * torch.conj(smaps), dim=1)

# ------------------------------------------------------------
# Diffusion schedule
# ------------------------------------------------------------

class DiffusionSchedule:
    def __init__(self, num_steps, beta_min=1e-4, beta_max=2e-2, device="cpu"):
        self.num_steps = num_steps
        self.device = device

        self.beta = torch.linspace(beta_min, beta_max, num_steps, device=device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(self, x0, t, noise):
        """
        Forward noising:
            x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) noise
        """
        a_bar = self.alpha_bar[t].view(-1, 1, 1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise

# ------------------------------------------------------------
# Time embedding
# ------------------------------------------------------------

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, t):
        """
        Sinusoidal-style embedding placeholder.
        t : (B,)
        """
        B = t.shape[0]
        emb = torch.stack([t.float(), t.float()**2], dim=-1)
        emb = F.pad(emb, (0, 30))  # placeholder fixed dimension
        emb = F.relu(self.lin1(emb))
        emb = F.relu(self.lin2(emb))
        return emb

# ------------------------------------------------------------
# Simple diffusion score network (U-Net style placeholder)
# ------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=32, time_dim=32):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)

        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 3, padding=1)
        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1)
        self.conv4 = nn.Conv2d(base_ch * 2, base_ch, 3, padding=1)
        self.out   = nn.Conv2d(base_ch, in_ch, 3, padding=1)

        self.fc_t1 = nn.Linear(time_dim, base_ch * 2)
        self.fc_t2 = nn.Linear(time_dim, base_ch)

    def forward(self, x, t):
        """
        x : (B, 2, H, W) real/imag channels
        t : (B,)
        """
        temb = self.time_embed(t)

        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))

        # inject time embedding
        t2 = self.fc_t1(temb).unsqueeze(-1).unsqueeze(-1)
        h2 = h2 + t2

        h3 = F.relu(self.conv3(h2))

        h4 = F.relu(self.conv4(h3))
        t4 = self.fc_t2(temb).unsqueeze(-1).unsqueeze(-1)
        h4 = h4 + t4

        out = self.out(h4)
        return out

# ------------------------------------------------------------
# Training loss: denoising score matching / epsilon prediction
# ------------------------------------------------------------

def diffusion_training_loss(model, schedule, x0):
    """
    x0 : (B, H, W) complex image
    model predicts noise epsilon
    """
    B = x0.shape[0]
    device = x0.device

    t = torch.randint(
        low=0, high=schedule.num_steps, size=(B,), device=device
    )
    noise = torch.randn_like(x0)

    x_t = schedule.q_sample(x0, t, noise)

    x_t_2ch = torch.stack([x_t.real, x_t.imag], dim=1)
    noise_2ch = torch.stack([noise.real, noise.imag], dim=1)

    noise_pred = model(x_t_2ch, t)

    loss = F.mse_loss(noise_pred, noise_2ch)
    stats = {
        "loss_total": float(loss.detach().cpu())
    }
    return loss, stats

# ------------------------------------------------------------
# Likelihood gradient for Gaussian MRI data model
# ------------------------------------------------------------

def likelihood_grad(x, y, mask, smaps=None, sigma=0.01):
    """
    Gradient of:
        (1 / 2 sigma^2) ||A(x) - y||^2
    """
    y_pred = forward_op(x, mask, smaps)
    resid = y_pred - y
    grad = adjoint_op(resid, mask, smaps) / (sigma**2)
    return grad

# ------------------------------------------------------------
# Posterior sampling / reconstruction
# ------------------------------------------------------------

class DiffusionMRIReconstructor:
    """
    Posterior inference using score prior + MRI likelihood.
    """
    def __init__(self, score_model, schedule, sigma=0.01,
                 dc_weight=1.0, step_scale=1.0):
        self.score_model = score_model
        self.schedule = schedule
        self.sigma = sigma
        self.dc_weight = dc_weight
        self.step_scale = step_scale

    def reconstruct(self, y, mask, smaps=None, num_steps=None):
        """
        Start from Gaussian noise and iteratively denoise while
        applying likelihood correction and data consistency.
        """
        device = y.device
        B = y.shape[0]

        if smaps is None:
            H, W = y.shape[-2], y.shape[-1]
        else:
            H, W = y.shape[-2], y.shape[-1]

        x = torch.randn(B, H, W, dtype=torch.cfloat, device=device)

        T = self.schedule.num_steps if num_steps is None else num_steps

        for t_idx in reversed(range(T)):
            t = torch.full((B,), t_idx, device=device, dtype=torch.long)

            x_2ch = torch.stack([x.real, x.imag], dim=1)
            eps_pred = self.score_model(x_2ch, t)
            eps_pred_c = torch.complex(eps_pred[:, 0], eps_pred[:, 1])

            beta_t = self.schedule.beta[t_idx]
            alpha_t = self.schedule.alpha[t_idx]
            alpha_bar_t = self.schedule.alpha_bar[t_idx]

            # reverse diffusion step (DDPM-style mean estimate)
            x = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred_c
            )

            # likelihood correction
            grad_like = likelihood_grad(
                x, y, mask, smaps=smaps, sigma=self.sigma
            )
            x = x - self.dc_weight * self.step_scale * grad_like

            # optional hard data consistency projection
            x = data_consistency_projection(x, y, mask, smaps=smaps)

            # inject stochasticity except at final step
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x

# ------------------------------------------------------------
# Optional posterior ensemble for uncertainty
# ------------------------------------------------------------

def posterior_ensemble(reconstructor, y, mask, smaps=None, num_samples=8):
    samples = []
    for _ in range(num_samples):
        x = reconstructor.reconstruct(y, mask, smaps=smaps)
        samples.append(x)
    samples = torch.stack(samples, dim=0)   # (S, B, H, W)

    mean = samples.mean(dim=0)
    var = ((samples - mean.unsqueeze(0)).abs()**2).mean(dim=0)

    return mean, var, samples

# ------------------------------------------------------------
# Example training loop for the diffusion prior
# ------------------------------------------------------------

def train_score_one_epoch(model, schedule, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        x0 = batch["image"].to(device)   # complex fully-sampled images

        optimizer.zero_grad()
        loss, stats = diffusion_training_loss(model, schedule, x0)
        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
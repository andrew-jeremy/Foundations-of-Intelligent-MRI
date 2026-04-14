
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Sampling design
# ------------------------------------------------------------

def variable_density_mask(shape, center_fraction=0.08, accel=4):
    """
    Simple example of a Cartesian variable-density mask.
    shape = (H, W)
    """
    H, W = shape
    mask = torch.zeros(H, W)

    # fully sample center
    num_center = int(W * center_fraction)
    start = (W - num_center) // 2
    end = start + num_center
    mask[:, start:end] = 1.0

    # random sampling outside center
    prob = min(1.0, 1.0 / accel)
    rand_mask = (torch.rand(H, W) < prob).float()
    mask = torch.maximum(mask, rand_mask)
    return mask

# ------------------------------------------------------------
# MRI operators
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

def forward_op(x, mask, smaps):
    """
    x     : (B, H, W) complex image
    mask  : (B, H, W) binary mask
    smaps : (B, C, H, W) complex coil sensitivity maps
    returns:
      y : (B, C, H, W) complex multicoil k-space
    """
    x_coils = x.unsqueeze(1) * smaps
    k = fft2c(x_coils)
    return mask.unsqueeze(1) * k

def adjoint_op(y, mask, smaps):
    """
    Adjoint operator A^* y
    """
    x_coils = ifft2c(mask.unsqueeze(1) * y)
    return torch.sum(x_coils * torch.conj(smaps), dim=1)

def hard_data_consistency(x, y, mask, smaps):
    """
    Replace acquired multicoil k-space values with measurements.
    """
    x_coils = x.unsqueeze(1) * smaps
    k_pred = fft2c(x_coils)
    k_dc = mask.unsqueeze(1) * y + (1.0 - mask.unsqueeze(1)) * k_pred
    x_dc_coils = ifft2c(k_dc)
    return torch.sum(x_dc_coils * torch.conj(smaps), dim=1)

# ------------------------------------------------------------
# Compressed sensing proximal-style block
# ------------------------------------------------------------

def soft_threshold(x, lam):
    mag = torch.abs(x)
    phase = x / (mag + 1e-12)
    mag_thresh = F.relu(mag - lam)
    return mag_thresh * phase

def wavelet_like_sparse_prox(x, lam):
    """
    Placeholder sparse proximal operator.
    In practice, use wavelets, finite differences, or learned transforms.
    """
    return soft_threshold(x, lam)

# ------------------------------------------------------------
# Learned prior / denoiser block
# ------------------------------------------------------------

class LearnedPriorBlock(nn.Module):
    """
    CNN denoiser / prior block operating on real-imag channels.
    """
    def __init__(self, in_ch=2, hidden_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, 3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# Neural operator-style correction block
# ------------------------------------------------------------

class SimpleNeuralOperatorBlock(nn.Module):
    """
    Placeholder neural operator-style correction.
    This uses global convolutions as a lightweight surrogate.
    In practice, replace with Fourier Neural Operator or related model.
    """
    def __init__(self, in_ch=2, hidden_ch=64):
        super().__init__()
        self.lift = nn.Conv2d(in_ch, hidden_ch, 1)
        self.global_conv = nn.Conv2d(hidden_ch, hidden_ch, 7, padding=3)
        self.proj = nn.Conv2d(hidden_ch, in_ch, 1)

    def forward(self, x):
        z = F.relu(self.lift(x))
        z = F.relu(self.global_conv(z))
        return self.proj(z)

# ------------------------------------------------------------
# One unrolled accelerated MRI stage
# ------------------------------------------------------------

class AccelerationStage(nn.Module):
    """
    One stage combining:
      1. data consistency
      2. compressed sensing proximal update
      3. learned prior correction
      4. neural-operator-style correction
    """
    def __init__(self, lam_cs=1e-3, hidden_ch=64):
        super().__init__()
        self.lam_cs = lam_cs
        self.prior_block = LearnedPriorBlock(in_ch=2, hidden_ch=hidden_ch)
        self.operator_block = SimpleNeuralOperatorBlock(
            in_ch=2, hidden_ch=hidden_ch
        )

    def forward(self, x, y, mask, smaps):
        # 1. physics-based data consistency
        x = hard_data_consistency(x, y, mask, smaps)

        # 2. compressed sensing proximal update
        x = wavelet_like_sparse_prox(x, self.lam_cs)

        # 3. learned prior correction
        x_2ch = torch.stack([x.real, x.imag], dim=1)
        delta_prior = self.prior_block(x_2ch)
        x_2ch = x_2ch + delta_prior

        # 4. neural operator correction
        delta_op = self.operator_block(x_2ch)
        x_2ch = x_2ch + delta_op

        x = torch.complex(x_2ch[:, 0], x_2ch[:, 1])
        return x

# ------------------------------------------------------------
# Full accelerated MRI system
# ------------------------------------------------------------

class OptimalAccelerationMRI(nn.Module):
    """
    Unified accelerated MRI reconstruction model.
    """
    def __init__(self, num_stages=8, lam_cs=1e-3, hidden_ch=64):
        super().__init__()
        self.stages = nn.ModuleList([
            AccelerationStage(lam_cs=lam_cs, hidden_ch=hidden_ch)
            for _ in range(num_stages)
        ])

    def forward(self, y, mask, smaps):
        """
        y     : (B, C, H, W) multicoil k-space
        mask  : (B, H, W)
        smaps : (B, C, H, W)
        """
        # Initial adjoint reconstruction
        x = adjoint_op(y, mask, smaps)
        intermediates = [x]

        for stage in self.stages:
            x = stage(x, y, mask, smaps)
            intermediates.append(x)

        return x, intermediates

# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------

def image_loss(x_hat, x_true):
    return F.mse_loss(x_hat.real, x_true.real) + \
           F.mse_loss(x_hat.imag, x_true.imag)

def kspace_loss(x_hat, y, mask, smaps):
    y_pred = forward_op(x_hat, mask, smaps)
    return torch.mean(torch.abs(y_pred - y)**2)

def perceptual_structure_loss(x_hat, x_true):
    """
    Simple gradient-based structure loss.
    """
    dx_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
    dy_hat = x_hat[:, :, 1:] - x_hat[:, :, :-1]

    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dy_true = x_true[:, :, 1:] - x_true[:, :, :-1]

    return (
        F.mse_loss(dx_hat.real, dx_true.real) +
        F.mse_loss(dx_hat.imag, dx_true.imag) +
        F.mse_loss(dy_hat.real, dy_true.real) +
        F.mse_loss(dy_hat.imag, dy_true.imag)
    )

def multi_stage_loss(intermediates, x_true, y, mask, smaps,
                     lambda_img=1.0, lambda_k=1.0,
                     lambda_struct=0.1, lambda_deep=0.3):
    x_final = intermediates[-1]

    loss_img = image_loss(x_final, x_true)
    loss_k = kspace_loss(x_final, y, mask, smaps)
    loss_struct = perceptual_structure_loss(x_final, x_true)

    loss_deep = 0.0
    for x_mid in intermediates[:-1]:
        loss_deep = loss_deep + image_loss(x_mid, x_true)
    loss_deep = loss_deep / max(len(intermediates) - 1, 1)

    total = (
        lambda_img * loss_img +
        lambda_k * loss_k +
        lambda_struct * loss_struct +
        lambda_deep * loss_deep
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_img": float(loss_img.detach().cpu()),
        "loss_kspace": float(loss_k.detach().cpu()),
        "loss_struct": float(loss_struct.detach().cpu()),
        "loss_deep": float(loss_deep.detach().cpu())
    }
    return total, stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        x_true = batch["target"].to(device)      # (B, H, W) complex
        smaps = batch["smaps"].to(device)        # (B, C, H, W) complex
        mask = batch["mask"].to(device)          # (B, H, W)
        y = batch["kspace"].to(device)           # (B, C, H, W) complex

        optimizer.zero_grad()

        x_hat, intermediates = model(y, mask, smaps)

        loss, stats = multi_stage_loss(
            intermediates=intermediates,
            x_true=x_true,
            y=y,
            mask=mask,
            smaps=smaps,
            lambda_img=1.0,
            lambda_k=1.0,
            lambda_struct=0.1,
            lambda_deep=0.3
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
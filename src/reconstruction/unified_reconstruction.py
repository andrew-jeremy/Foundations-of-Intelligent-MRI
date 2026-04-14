import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# MRI forward / adjoint operators
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

def hard_data_consistency(x, y, mask, smaps=None):
    """
    Replace predicted k-space values at sampled locations with measured data.
    """
    if smaps is None:
        k_pred = fft2c(x)
        k_dc = mask * y + (1.0 - mask) * k_pred
        return ifft2c(k_dc)
    else:
        x_coils = x.unsqueeze(1) * smaps
        k_pred = fft2c(x_coils)
        k_dc = mask.unsqueeze(1) * y + (1.0 - mask.unsqueeze(1)) * k_pred
        x_dc = ifft2c(k_dc)
        return torch.sum(x_dc * torch.conj(smaps), dim=1)

# ------------------------------------------------------------
# Basic denoising / artifact suppression block
# ------------------------------------------------------------

class RefinementBlock(nn.Module):
    """
    Learned image-domain artifact correction and denoising block.
    Input/output are 2-channel real/imag images.
    """
    def __init__(self, in_ch=2, hidden_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# Optional generative prior block (placeholder)
# ------------------------------------------------------------

class PriorBlock(nn.Module):
    """
    Learned prior correction block.
    Could represent a diffusion denoiser, score block, or adversarial prior.
    Here we implement a lightweight residual CNN as a placeholder.
    """
    def __init__(self, in_ch=2, hidden_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------
# One unrolled stage
# ------------------------------------------------------------

class ReconstructionStage(nn.Module):
    """
    One stage of unified reconstruction:
      1. data consistency
      2. artifact correction / denoising
      3. prior correction
    """
    def __init__(self, hidden_ch=64):
        super().__init__()
        self.refine = RefinementBlock(in_ch=2, hidden_ch=hidden_ch)
        self.prior  = PriorBlock(in_ch=2, hidden_ch=hidden_ch)

    def forward(self, x, y, mask, smaps=None):
        # Physics-based data consistency
        x = hard_data_consistency(x, y, mask, smaps)

        # Convert to real/imag channels
        x_2ch = torch.stack([x.real, x.imag], dim=1)

        # Learned artifact suppression
        delta_refine = self.refine(x_2ch)
        x_2ch = x_2ch + delta_refine

        # Learned prior correction
        delta_prior = self.prior(x_2ch)
        x_2ch = x_2ch + delta_prior

        # Back to complex image
        x = torch.complex(x_2ch[:, 0], x_2ch[:, 1])

        return x

# ------------------------------------------------------------
# Unified reconstruction network
# ------------------------------------------------------------

class UnifiedMRIReconstructor(nn.Module):
    """
    Multi-stage unrolled unified MRI reconstruction framework.
    """
    def __init__(self, num_stages=6, hidden_ch=64):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([
            ReconstructionStage(hidden_ch=hidden_ch)
            for _ in range(num_stages)
        ])

    def forward(self, y, mask, smaps=None):
        """
        y     : measured k-space
        mask  : sampling mask
        smaps : optional coil sensitivity maps
        """
        # Initial adjoint / zero-filled reconstruction
        x = adjoint_op(y, mask, smaps)

        # Unrolled iterative reconstruction
        intermediates = [x]
        for stage in self.stages:
            x = stage(x, y, mask, smaps)
            intermediates.append(x)

        return x, intermediates

# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------

def image_loss(x_hat, x_true):
    """
    Basic supervised image loss.
    """
    return F.mse_loss(x_hat.real, x_true.real) + \
           F.mse_loss(x_hat.imag, x_true.imag)

def kspace_loss(x_hat, y, mask, smaps=None):
    """
    Data-consistency loss in measurement space.
    """
    y_pred = forward_op(x_hat, mask, smaps)
    return torch.mean(torch.abs(y_pred - y)**2)

def edge_preserving_loss(x_hat, x_true):
    """
    Simple gradient-domain loss to encourage structural fidelity.
    """
    dx_hat = x_hat[:, 1:, :] - x_hat[:, :-1, :]
    dy_hat = x_hat[:, :, 1:] - x_hat[:, :, :-1]

    dx_true = x_true[:, 1:, :] - x_true[:, :-1, :]
    dy_true = x_true[:, :, 1:] - x_true[:, :, :-1]

    loss_dx = F.mse_loss(dx_hat.real, dx_true.real) + \
              F.mse_loss(dx_hat.imag, dx_true.imag)
    loss_dy = F.mse_loss(dy_hat.real, dy_true.real) + \
              F.mse_loss(dy_hat.imag, dy_true.imag)

    return loss_dx + loss_dy

def multi_stage_loss(intermediates, x_true, y, mask, smaps=None,
                     lambda_img=1.0, lambda_k=1.0,
                     lambda_edge=0.1, lambda_deep=0.5):
    """
    Unified training objective combining:
      - final image fidelity
      - measurement consistency
      - edge/structure preservation
      - deep supervision across stages
    """
    x_final = intermediates[-1]

    loss_img = image_loss(x_final, x_true)
    loss_k = kspace_loss(x_final, y, mask, smaps)
    loss_edge = edge_preserving_loss(x_final, x_true)

    # Deep supervision over intermediate reconstructions
    loss_deep = 0.0
    for x_mid in intermediates[:-1]:
        loss_deep = loss_deep + image_loss(x_mid, x_true)
    loss_deep = loss_deep / max(len(intermediates) - 1, 1)

    total = (
        lambda_img * loss_img +
        lambda_k * loss_k +
        lambda_edge * loss_edge +
        lambda_deep * loss_deep
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_img": float(loss_img.detach().cpu()),
        "loss_kspace": float(loss_k.detach().cpu()),
        "loss_edge": float(loss_edge.detach().cpu()),
        "loss_deep": float(loss_deep.detach().cpu())
            if not isinstance(loss_deep, float) else loss_deep
    }

    return total, stats

# ------------------------------------------------------------
# Optional uncertainty head (placeholder)
# ------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """
    Optional uncertainty estimator from final reconstruction features.
    """
    def __init__(self, in_ch=2, hidden_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_ch, 1, 3, padding=1),
            nn.Softplus()
        )

    def forward(self, x):
        """
        x : complex image -> variance map
        """
        x_2ch = torch.stack([x.real, x.imag], dim=1)
        return self.net(x_2ch)

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        y = batch["kspace"].to(device)
        mask = batch["mask"].to(device)
        x_true = batch["target"].to(device)
        smaps = batch.get("smaps", None)

        if smaps is not None:
            smaps = smaps.to(device)

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
            lambda_edge=0.1,
            lambda_deep=0.5
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running

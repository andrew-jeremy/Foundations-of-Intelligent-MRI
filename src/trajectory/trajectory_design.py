import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------
# Trajectory parameterization
# --------------------------------------------------------

class LearnableTrajectory(nn.Module):
    """
    Learnable 2D k-space trajectory represented by control points.

    Parameters:
      num_samples : number of trajectory samples
      init_scale  : initialization scale in normalized k-space
    """
    def __init__(self, num_samples, init_scale=0.25):
        super().__init__()
        self.num_samples = num_samples

        # learnable k-space points: (T, 2)
        self.k_points = nn.Parameter(
            init_scale * torch.randn(num_samples, 2)
        )

    def forward(self):
        """
        Returns trajectory points in normalized k-space.
        shape: (T, 2)
        """
        return self.k_points

# ---------------------------------------------------------
# Physical constraint penalties
# ---------------------------------------------------------

def gradient_amplitude_penalty(k_points, gmax):
    """
    Penalize large gradient amplitude, approximated by trajectory speed.
    """
    dk = k_points[1:] - k_points[:-1]              # (T-1, 2)
    speed = torch.sqrt(torch.sum(dk**2, dim=-1) + 1e-12)
    return F.relu(speed - gmax).pow(2).mean()

def slew_rate_penalty(k_points, smax):
    """
    Penalize large slew rate, approximated by second differences.
    """
    dk = k_points[1:] - k_points[:-1]
    d2k = dk[1:] - dk[:-1]                         # (T-2, 2)
    accel = torch.sqrt(torch.sum(d2k**2, dim=-1) + 1e-12)
    return F.relu(accel - smax).pow(2).mean()

def kspace_extent_penalty(k_points, kmax):
    """
    Penalize points outside allowable k-space extent.
    """
    radius = torch.sqrt(torch.sum(k_points**2, dim=-1) + 1e-12)
    return F.relu(radius - kmax).pow(2).mean()

# ----------------------------------------------------------
# Approximate differentiable NUFFT-like sampler
# ----------------------------------------------------------

def differentiable_fourier_sample(x, k_points):
    """
    Approximate nonuniform Fourier sampling of image x at k_points.

    x        : (B, H, W) real or complex image
    k_points : (T, 2) normalized coordinates in [-0.5, 0.5]

    This is schematic pseudocode. In practice, one would use a true
    differentiable NUFFT implementation.
    """
    B, H, W = x.shape
    device = x.device

    # spatial coordinate grid
    yy, xx = torch.meshgrid(
        torch.linspace(-0.5, 0.5, H, device=device),
        torch.linspace(-0.5, 0.5, W, device=device),
        indexing='ij'
    )

    xx = xx.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    yy = yy.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    x = x.unsqueeze(1)                  # (B,1,H,W)

    kx = k_points[:, 0].view(1, -1, 1, 1)   # (1,T,1,1)
    ky = k_points[:, 1].view(1, -1, 1, 1)   # (1,T,1,1)

    phase = torch.exp(
        -1j * 2.0 * torch.pi * (kx * xx + ky * yy)
    )                                         # (1,T,H,W)

    samples = torch.sum(x * phase, dim=(-2, -1))  # (B,T)
    return samples

# --------------------------------------------------------
# Simple reconstruction network
# --------------------------------------------------------

class ReconstructionNet(nn.Module):
    """
    Simple decoder from sampled k-space values to image.
    This is schematic; in practice use a stronger inverse model.
    """
    def __init__(self, num_samples, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.net = nn.Sequential(
            nn.Linear(num_samples * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, H * W)
        )

    def forward(self, y):
        """
        y : (B,T) complex sampled k-space
        """
        y_feat = torch.cat([y.real, y.imag], dim=-1)  # (B, 2T)
        out = self.net(y_feat)
        return out.view(y.shape[0], self.H, self.W)

# ---------------------------------------------------------
# Joint acquisition-reconstruction model
# ---------------------------------------------------------

class TrajectoryOptimizedMRI(nn.Module):
    """
    Joint trajectory + reconstruction model.
    """
    def __init__(self, num_samples, H, W):
        super().__init__()
        self.trajectory = LearnableTrajectory(num_samples)
        self.reconstructor = ReconstructionNet(num_samples, H, W)

    def forward(self, x_true):
        """
        x_true : (B,H,W)
        """
        k_points = self.trajectory()                     # (T,2)
        y = differentiable_fourier_sample(x_true, k_points)
        x_hat = self.reconstructor(y)
        return x_hat, y, k_points

# -------------------------------------------------------
# Loss function
# -------------------------------------------------------

def trajectory_design_loss(
    x_hat, x_true, k_points,
    lambda_grad=1e-2,
    lambda_slew=1e-2,
    lambda_extent=1e-2,
    gmax=0.05,
    smax=0.02,
    kmax=0.5
):
    """
    End-to-end trajectory design loss.
    """
    loss_recon = F.mse_loss(x_hat, x_true)

    loss_grad = gradient_amplitude_penalty(k_points, gmax)
    loss_slew = slew_rate_penalty(k_points, smax)
    loss_extent = kspace_extent_penalty(k_points, kmax)

    total = (
        loss_recon +
        lambda_grad * loss_grad +
        lambda_slew * loss_slew +
        lambda_extent * loss_extent
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_recon": float(loss_recon.detach().cpu()),
        "loss_grad": float(loss_grad.detach().cpu()),
        "loss_slew": float(loss_slew.detach().cpu()),
        "loss_extent": float(loss_extent.detach().cpu())
    }

    return total, stats

# -------------------------------------------------------
# Example training loop
# -------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        x_true = batch["image"].to(device)   # (B,H,W)

        optimizer.zero_grad()

        x_hat, y, k_points = model(x_true)

        loss, stats = trajectory_design_loss(
            x_hat, x_true, k_points,
            lambda_grad=1e-2,
            lambda_slew=1e-2,
            lambda_extent=1e-2,
            gmax=0.05,
            smax=0.02,
            kmax=0.5
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
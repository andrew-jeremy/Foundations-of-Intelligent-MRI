import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------
# Basic MRI operators
# ------------------------------------------------

def fft2c(x):
    # centered 2D FFT, placeholder
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

def ifft2c(k):
    # centered 2D IFFT, placeholder
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1)
    )

def forward_op(x, mask, smaps=None):
    """
    x     : (B, T, H, W) complex image sequence
    mask  : (B, T, H, W) binary sampling mask
    smaps : optional coil sensitivity maps (B, C, H, W)
    returns undersampled k-space
    """
    if smaps is None:
        k = fft2c(x)
        return mask * k
    else:
        x_coils = x.unsqueeze(2) * smaps.unsqueeze(1)   # (B,T,C,H,W)
        k = fft2c(x_coils)
        return mask.unsqueeze(2) * k

def adjoint_op(k, mask, smaps=None):
    """
    Adjoint of forward operator.
    """
    if smaps is None:
        return ifft2c(mask * k)
    else:
        x_coils = ifft2c(mask.unsqueeze(2) * k)
        return torch.sum(
            x_coils * torch.conj(smaps.unsqueeze(1)),
            dim=2
        )

def data_consistency(x_pred, y, mask, smaps=None):
    """
    Enforce consistency with acquired k-space samples.
    """
    k_pred = forward_op(x_pred, torch.ones_like(mask), smaps)
    if smaps is None:
        k_dc = mask * y + (1.0 - mask) * k_pred
    else:
        k_dc = mask.unsqueeze(2) * y + (1.0 - mask.unsqueeze(2)) * k_pred
    return adjoint_op(k_dc, torch.ones_like(mask), smaps)

# ----------------------------------------------
# Low-rank dynamic model
# ----------------------------------------------

class TemporalLowRankModule(nn.Module):
    """
    Learns latent temporal coefficients constrained to low rank.
    Input:
        z : (B, T, r)
    Output:
        z_lowrank : (B, T, r)
    """
    def __init__(self, r):
        super().__init__()
        self.temporal_net = nn.Sequential(
            nn.Linear(r, 2 * r),
            nn.ReLU(),
            nn.Linear(2 * r, r)
        )

    def forward(self, z):
        B, T, r = z.shape
        z_out = self.temporal_net(z.reshape(B * T, r))
        z_out = z_out.reshape(B, T, r)
        return z_out

class SpatialDecoder(nn.Module):
    """
    Decode latent vector into image frame.
    """
    def __init__(self, r, H, W):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(r, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, H * W * 2)  # real + imag
        )
        self.H = H
        self.W = W

    def forward(self, z):
        out = self.fc(z)
        out = out.view(z.shape[0], self.H, self.W, 2)
        re = out[..., 0]
        im = out[..., 1]
        return torch.complex(re, im)

class LowRankDynamicMRIModel(nn.Module):
    """
    End-to-end low-rank dynamic MRI model.
    Input:
        zero-filled image sequence
    Output:
        reconstructed dynamic image sequence
    """
    def __init__(self, T, H, W, r):
        super().__init__()
        self.T = T
        self.H = H
        self.W = W
        self.r = r

        # framewise encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, r)
        )

        self.temporal_lowrank = TemporalLowRankModule(r)
        self.decoder = SpatialDecoder(r, H, W)

        # optional refinement CNN
        self.refine = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1)
        )

    def forward(self, x_zf):
        """
        x_zf: (B, T, H, W) complex zero-filled reconstruction
        """
        B, T, H, W = x_zf.shape

        # encode each frame into latent code
        z_list = []
        for t in range(T):
            xt = x_zf[:, t]
            xt_2ch = torch.stack([xt.real, xt.imag], dim=1)  # (B,2,H,W)
            zt = self.encoder(xt_2ch)
            z_list.append(zt)
        z = torch.stack(z_list, dim=1)  # (B,T,r)

        # low-rank temporal projection / modeling
        z_lr = self.temporal_lowrank(z)

        # decode frames
        x_rec_list = []
        for t in range(T):
            xt = self.decoder(z_lr[:, t])   # complex image
            xt_2ch = torch.stack([xt.real, xt.imag], dim=1)
            delta = self.refine(xt_2ch)
            xt_ref = torch.complex(
                xt.real + delta[:, 0],
                xt.imag + delta[:, 1]
            )
            x_rec_list.append(xt_ref)

        x_rec = torch.stack(x_rec_list, dim=1)  # (B,T,H,W)
        return x_rec, z_lr

# ---------------------------------------------
# Low-rank losses
# ---------------------------------------------

def temporal_smoothness_loss(z):
    """
    z : (B, T, r)
    """
    return ((z[:, 1:] - z[:, :-1])**2).mean()

def nuclear_norm_proxy(z):
    """
    z : (B, T, r)
    Low-rank regularization via singular values of latent matrix.
    """
    loss = 0.0
    for b in range(z.shape[0]):
        s = torch.linalg.svdvals(z[b])  # singular values of (T,r)
        loss = loss + s.sum()
    return loss / z.shape[0]

def reconstruction_loss(x_rec, x_gt):
    return F.mse_loss(x_rec.real, x_gt.real) + \
           F.mse_loss(x_rec.imag, x_gt.imag)

def kspace_data_loss(x_rec, y, mask, smaps=None):
    y_pred = forward_op(x_rec, mask, smaps)
    return torch.mean(torch.abs(y_pred - y)**2)

# ---------------------------------------------
# Full training objective
# ---------------------------------------------

def dynamic_mri_loss(
    x_rec, z_lr, y, mask,
    x_gt=None, smaps=None,
    lambda_k=1.0,
    lambda_img=1.0,
    lambda_rank=1e-3,
    lambda_temp=1e-3
):
    loss_k = kspace_data_loss(x_rec, y, mask, smaps)
    loss_img = 0.0
    if x_gt is not None:
        loss_img = reconstruction_loss(x_rec, x_gt)

    loss_rank = nuclear_norm_proxy(z_lr)
    loss_temp = temporal_smoothness_loss(z_lr)

    total = (
        lambda_k * loss_k +
        lambda_img * loss_img +
        lambda_rank * loss_rank +
        lambda_temp * loss_temp
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_kspace": float(loss_k.detach().cpu()),
        "loss_image": float(loss_img.detach().cpu()) \
            if not isinstance(loss_img, float) else loss_img,
        "loss_rank": float(loss_rank.detach().cpu()),
        "loss_temp": float(loss_temp.detach().cpu())
    }

    return total, stats

# ------------------------------------------------
# Example training loop
# ------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        y = batch["kspace"].to(device)          # measured k-space
        mask = batch["mask"].to(device)         # sampling mask
        x_zf = batch["zf_image"].to(device)     # zero-filled image
        x_gt = batch.get("target", None)
        smaps = batch.get("smaps", None)

        if x_gt is not None:
            x_gt = x_gt.to(device)
        if smaps is not None:
            smaps = smaps.to(device)

        optimizer.zero_grad()

        x_rec, z_lr = model(x_zf)

        # optional data-consistency correction
        x_rec = data_consistency(x_rec, y, mask, smaps)

        loss, stats = dynamic_mri_loss(
            x_rec, z_lr, y, mask,
            x_gt=x_gt, smaps=smaps,
            lambda_k=1.0,
            lambda_img=1.0,
            lambda_rank=1e-3,
            lambda_temp=1e-3
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running


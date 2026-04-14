
import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_matvec(B_re, B_im, w_re, w_im):
    out_re = torch.einsum('bnc,bc->bn', B_re, w_re) \
           - torch.einsum('bnc,bc->bn', B_im, w_im)
    out_im = torch.einsum('bnc,bc->bn', B_re, w_im) \
           + torch.einsum('bnc,bc->bn', B_im, w_re)
    return out_re, out_im

def complex_abs(re, im, eps=1e-12):
    return torch.sqrt(re**2 + im**2 + eps)

def quadratic_form_complex(Q_re, Q_im, w_re, w_im):
    w = torch.complex(w_re, w_im)
    Q = torch.complex(Q_re, Q_im)
    val = torch.einsum('bc,bcd,bd->b', torch.conj(w), Q, w)
    return torch.real(val)

def hinge_square(x):
    return F.relu(x)**2

class ShimNet(nn.Module):
    def __init__(self, in_ch, M, C):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.head_b0   = nn.Linear(128, M)
        self.head_w_re = nn.Linear(128, C)
        self.head_w_im = nn.Linear(128, C)

    def forward(self, x):
        z = self.encoder(x)
        z = self.fc(z)
        u    = self.head_b0(z)
        w_re = self.head_w_re(z)
        w_im = self.head_w_im(z)
        return u, w_re, w_im

def project_global_power(w_re, w_im, Pmax, eps=1e-12):
    norm2 = torch.sum(w_re**2 + w_im**2, dim=-1, keepdim=True)
    scale = torch.sqrt(torch.clamp(Pmax / (norm2 + eps), max=1.0))
    return w_re * scale, w_im * scale

def shim_loss(
    u, w_re, w_im,
    d_b0, S_b0, mask_b0,
    B1_re, B1_im, target_b1, mask_b1,
    Q_list_re, Q_list_im, sar_limits,
    P_re, P_im, power_limit,
    shim_limits=None,
    u_star=None, w_star_re=None, w_star_im=None,
    lambda_b0=1.0, lambda_b1=1.0,
    lambda_u_reg=1e-4, lambda_power=1.0,
    lambda_sar=1.0, lambda_sup=0.1
):
    Su = torch.einsum('bnm,bm->bn', S_b0, u)
    r_b0 = d_b0 + Su
    loss_b0 = torch.sum(mask_b0 * (r_b0**2), dim=-1).mean()
    loss_u_reg = torch.sum(u**2, dim=-1).mean()

    loss_hw = 0.0
    if shim_limits is not None:
        loss_hw = hinge_square(torch.abs(u) - shim_limits).mean()

    b1_re, b1_im = complex_matvec(B1_re, B1_im, w_re, w_im)
    b1_mag = complex_abs(b1_re, b1_im)
    loss_b1 = torch.sum(
        mask_b1 * ((b1_mag - target_b1)**2), dim=-1
    ).mean()

    power_val = quadratic_form_complex(P_re, P_im, w_re, w_im)
    loss_power = hinge_square(power_val - power_limit).mean()

    loss_sar = 0.0
    for Q_re, Q_im, s_lim in zip(Q_list_re, Q_list_im, sar_limits):
        sar_val = quadratic_form_complex(Q_re, Q_im, w_re, w_im)
        loss_sar = loss_sar + hinge_square(sar_val - s_lim).mean()

    loss_sup = 0.0
    if u_star is not None:
        loss_sup = loss_sup + F.mse_loss(u, u_star)
    if w_star_re is not None and w_star_im is not None:
        loss_sup = loss_sup + F.mse_loss(w_re, w_star_re)
        loss_sup = loss_sup + F.mse_loss(w_im, w_star_im)

    total = (
        lambda_b0 * loss_b0 +
        lambda_b1 * loss_b1 +
        lambda_u_reg * loss_u_reg +
        lambda_power * loss_power +
        lambda_sar * loss_sar +
        lambda_sup * loss_sup +
        loss_hw
    )

    return total

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    for batch in loader:
        x = batch["input"].to(device)
        d_b0 = batch["d_b0"].to(device)
        S_b0 = batch["S_b0"].to(device)
        mask_b0 = batch["mask_b0"].to(device)

        B1_re = batch["B1_re"].to(device)
        B1_im = batch["B1_im"].to(device)
        target_b1 = batch["target_b1"].to(device)
        mask_b1 = batch["mask_b1"].to(device)

        Q_list_re = [q.to(device) for q in batch["Q_list_re"]]
        Q_list_im = [q.to(device) for q in batch["Q_list_im"]]
        sar_limits = batch["sar_limits"]

        P_re = batch["P_re"].to(device)
        P_im = batch["P_im"].to(device)
        power_limit = batch["power_limit"].to(device)

        optimizer.zero_grad()
        u, w_re, w_im = model(x)
        w_re, w_im = project_global_power(
            w_re, w_im, power_limit.unsqueeze(-1)
        )

        loss = shim_loss(
            u, w_re, w_im,
            d_b0, S_b0, mask_b0,
            B1_re, B1_im, target_b1, mask_b1,
            Q_list_re, Q_list_im, sar_limits,
            P_re, P_im, power_limit
        )

        loss.backward()
        optimizer.step()
\end{lstlisting}
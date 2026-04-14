import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Example effective field model
# ------------------------------------------------------------

class EffectiveFieldModel(nn.Module):
    """
    Defines Bx(t), By(t), Bz(t) for a given pulse sequence.
    This may be replaced by a more detailed RF/gradient model.
    """
    def __init__(self, gamma, B0=3.0, B1=1e-4, omega_rf=0.0):
        super().__init__()
        self.gamma = gamma
        self.B0 = B0
        self.B1 = B1
        self.omega_rf = omega_rf

    def forward(self, t):
        """
        t : (N,1)
        returns Bx, By, Bz each of shape (N,1)
        Example: rotating RF field plus static B0
        """
        Bx = self.B1 * torch.cos(self.omega_rf * t)
        By = self.B1 * torch.sin(self.omega_rf * t)
        Bz = self.B0 * torch.ones_like(t)
        return Bx, By, Bz

# ------------------------------------------------------------
# PINN for Bloch dynamics
# ------------------------------------------------------------

class BlochPINN(nn.Module):
    """
    Neural network approximating magnetization trajectory:
        t -> (Mx, My, Mz)
    Optional conditioning variables may be concatenated to t.
    """
    def __init__(self, in_dim=1, hidden_dim=128, depth=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        x : (N, in_dim)
        returns M : (N, 3)
        """
        return self.net(x)

# ------------------------------------------------------------
# Autograd derivatives
# ------------------------------------------------------------

def time_derivative(y, t):
    """
    Compute dy/dt using PyTorch autograd.
    y : (N,1)
    t : (N,1), requires_grad=True
    """
    grad = torch.autograd.grad(
        y, t,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad

# ------------------------------------------------------------
# Bloch residuals
# ------------------------------------------------------------

def bloch_residuals(model, t, field_model, gamma, T1, T2, M0, cond=None):
    """
    Compute PINN residuals for Bloch equations.

    t    : (N,1), requires_grad=True
    cond : optional conditioning tensor (N, d)
    """
    if cond is not None:
        inp = torch.cat([t, cond], dim=-1)
    else:
        inp = t

    M = model(inp)        # (N,3)
    Mx = M[:, 0:1]
    My = M[:, 1:2]
    Mz = M[:, 2:3]

    dMx_dt = time_derivative(Mx, t)
    dMy_dt = time_derivative(My, t)
    dMz_dt = time_derivative(Mz, t)

    Bx, By, Bz = field_model(t)

    Rx = dMx_dt - gamma * (My * Bz - Mz * By) + Mx / T2
    Ry = dMy_dt - gamma * (Mz * Bx - Mx * Bz) + My / T2
    Rz = dMz_dt - gamma * (Mx * By - My * Bx) + (Mz - M0) / T1

    return Rx, Ry, Rz, M

# ------------------------------------------------------------
# Optional MRI signal model
# ------------------------------------------------------------

def transverse_signal(M):
    """
    Example signal model:
        s(t) = Mx(t) + i My(t)
    """
    Mx = M[:, 0]
    My = M[:, 1]
    return torch.complex(Mx, My)

# ------------------------------------------------------------
# PINN loss
# ------------------------------------------------------------

def bloch_pinn_loss(
    model,
    t_colloc,
    t_ic,
    M_ic,
    field_model,
    gamma,
    T1,
    T2,
    M0,
    cond_colloc=None,
    cond_ic=None,
    t_data=None,
    s_data=None,
    cond_data=None,
    lambda_phys=1.0,
    lambda_ic=1.0,
    lambda_data=0.0
):
    """
    Physics-informed loss:
      - Bloch residual loss
      - initial condition loss
      - optional signal data loss
    """

    # Physics residuals
    Rx, Ry, Rz, M_colloc = bloch_residuals(
        model, t_colloc, field_model, gamma, T1, T2, M0,
        cond=cond_colloc
    )
    loss_phys = torch.mean(Rx**2) + torch.mean(Ry**2) + torch.mean(Rz**2)

    # Initial condition loss
    if cond_ic is not None:
        inp_ic = torch.cat([t_ic, cond_ic], dim=-1)
    else:
        inp_ic = t_ic
    M_pred_ic = model(inp_ic)
    loss_ic = F.mse_loss(M_pred_ic, M_ic)

    # Optional signal data loss
    loss_data = 0.0
    if t_data is not None and s_data is not None:
        if cond_data is not None:
            inp_data = torch.cat([t_data, cond_data], dim=-1)
        else:
            inp_data = t_data
        M_data = model(inp_data)
        s_pred = transverse_signal(M_data)

        loss_data = F.mse_loss(s_pred.real, s_data.real) + \
                    F.mse_loss(s_pred.imag, s_data.imag)

    total = (
        lambda_phys * loss_phys +
        lambda_ic * loss_ic +
        lambda_data * loss_data
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_phys": float(loss_phys.detach().cpu()),
        "loss_ic": float(loss_ic.detach().cpu()),
        "loss_data": float(loss_data.detach().cpu())
            if not isinstance(loss_data, float) else loss_data
    }

    return total, stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_bloch_pinn_one_epoch(
    model,
    field_model,
    optimizer,
    device,
    num_colloc=1024,
    t_min=0.0,
    t_max=5e-3,
    gamma=2.675e8,
    T1=1.0,
    T2=0.1,
    M0=1.0,
    lambda_phys=1.0,
    lambda_ic=10.0,
    lambda_data=0.0,
    measured_signal=None
):
    model.train()
    optimizer.zero_grad()

    # Collocation points in time
    t_colloc = torch.rand(num_colloc, 1, device=device) * (t_max - t_min) + t_min
    t_colloc.requires_grad_(True)

    # Initial condition at t = 0
    t_ic = torch.zeros(1, 1, device=device, requires_grad=True)
    M_ic = torch.tensor([[0.0, 0.0, M0]], device=device)

    # Optional data samples
    t_data = None
    s_data = None
    if measured_signal is not None:
        t_data = measured_signal["t"].to(device)
        t_data.requires_grad_(True)
        s_data = measured_signal["s"].to(device)

    loss, stats = bloch_pinn_loss(
        model=model,
        t_colloc=t_colloc,
        t_ic=t_ic,
        M_ic=M_ic,
        field_model=field_model,
        gamma=gamma,
        T1=T1,
        T2=T2,
        M0=M0,
        t_data=t_data,
        s_data=s_data,
        lambda_phys=lambda_phys,
        lambda_ic=lambda_ic,
        lambda_data=lambda_data
    )

    loss.backward()
    optimizer.step()

    return stats

# ------------------------------------------------------------
# Example inference / simulation
# ------------------------------------------------------------

def evaluate_magnetization(model, device, t_min=0.0, t_max=5e-3, num_points=1000):
    model.eval()
    t = torch.linspace(t_min, t_max, num_points, device=device).view(-1, 1)
    with torch.no_grad():
        M = model(t)
    return t, M
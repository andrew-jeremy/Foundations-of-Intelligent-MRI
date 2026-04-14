
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------
# Example forward model
# -------------------------------------------------------

class MRSignalModel(nn.Module):
    """
    Generic MRI forward model F(theta).

    Example shown here:
        inversion recovery / relaxation-style model
    with parameters:
        theta[...,0] = M0
        theta[...,1] = T1
    and measurements at inversion times TI.

    This can be replaced by any differentiable MRI signal model.
    """
    def __init__(self, TI):
        super().__init__()
        self.register_buffer("TI", TI)   # (m,)

    def forward(self, theta):
        """
        theta : (B, p)
        returns predicted signal : (B, m)
        """
        M0 = theta[:, 0:1]
        T1 = theta[:, 1:2]

        # enforce positivity softly if desired
        T1 = F.softplus(T1) + 1e-6

        TI = self.TI.unsqueeze(0)  # (1,m)
        signal = M0 * (1.0 - 2.0 * torch.exp(-TI / T1))
        return signal

# ------------------------------------------------------
# Gaussian likelihood
# ------------------------------------------------------

def gaussian_neg_log_likelihood(y, y_pred, sigma):
    """
    Negative log-likelihood up to additive constant:
        (1 / 2 sigma^2) ||y - y_pred||^2
    """
    return torch.mean(torch.sum((y - y_pred)**2, dim=-1) / (2.0 * sigma**2))

# ------------------------------------------------------
# Gaussian prior
# ------------------------------------------------------

class GaussianPrior(nn.Module):
    """
    p(theta) = N(mu, Sigma)
    Assume diagonal covariance for simplicity.
    """
    def __init__(self, mu, var):
        super().__init__()
        self.register_buffer("mu", mu)     # (p,)
        self.register_buffer("var", var)   # (p,)

    def neg_log_prob(self, theta):
        """
        Negative log-prior up to additive constant:
            0.5 * sum ((theta - mu)^2 / var)
        """
        diff = theta - self.mu.unsqueeze(0)
        return torch.mean(0.5 * torch.sum(diff**2 / self.var.unsqueeze(0), dim=-1))

# ---------------------------------------------------------
# Optional log-normal prior for positive parameters
# ---------------------------------------------------------

class LogNormalPrior(nn.Module):
    """
    Log-normal prior on selected positive parameters.
    Assume all parameters here are positive for simplicity.
    """
    def __init__(self, mu_log, var_log):
        super().__init__()
        self.register_buffer("mu_log", mu_log)
        self.register_buffer("var_log", var_log)

    def neg_log_prob(self, theta):
        theta_pos = F.softplus(theta) + 1e-6
        log_theta = torch.log(theta_pos)
        diff = log_theta - self.mu_log.unsqueeze(0)
        return torch.mean(
            0.5 * torch.sum(diff**2 / self.var_log.unsqueeze(0), dim=-1)
        )

# ------------------------------------------------------------
# Spatial smoothness prior (optional for voxelwise parameter maps)
# ------------------------------------------------------------

def spatial_smoothness_prior(theta_map):
    """
    theta_map : (B, P, H, W)
    Penalize finite differences across spatial neighbors.
    """
    dx = theta_map[:, :, 1:, :] - theta_map[:, :, :-1, :]
    dy = theta_map[:, :, :, 1:] - theta_map[:, :, :, :-1]
    return torch.mean(dx**2) + torch.mean(dy**2)

# ----------------------------------------------------
# Posterior energy
# ----------------------------------------------------

def posterior_energy(theta, y, signal_model, prior, sigma):
    """
    Negative log-posterior up to additive constant.
    """
    y_pred = signal_model(theta)
    nll = gaussian_neg_log_likelihood(y, y_pred, sigma)
    nlp = prior.neg_log_prob(theta)
    return nll + nlp

# ----------------------------------------------------
# MAP estimation
# ----------------------------------------------------

class MAPEstimator:
    """
    Optimize theta by minimizing negative log-posterior.
    """
    def __init__(self, signal_model, prior, sigma=0.01, lr=1e-2, num_steps=200):
        self.signal_model = signal_model
        self.prior = prior
        self.sigma = sigma
        self.lr = lr
        self.num_steps = num_steps

    def solve(self, y, theta_init):
        """
        y         : (B, m)
        theta_init: (B, p)
        """
        theta = theta_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([theta], lr=self.lr)

        history = []
        for _ in range(self.num_steps):
            optimizer.zero_grad()
            energy = posterior_energy(
                theta, y, self.signal_model, self.prior, self.sigma
            )
            energy.backward()
            optimizer.step()
            history.append(float(energy.detach().cpu()))

        return theta.detach(), history

# ----------------------------------------------------
# Variational Bayesian inference
# ----------------------------------------------------

class VariationalPosterior(nn.Module):
    """
    Mean-field Gaussian posterior approximation:
        q(theta|y) = N(mu_q(y), diag(var_q(y)))
    learned from measurements y using an inference network.
    """
    def __init__(self, input_dim, param_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.mu_head = nn.Linear(128, param_dim)
        self.logvar_head = nn.Linear(128, param_dim)

    def forward(self, y):
        h = self.net(y)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

def kl_gaussian(mu_q, logvar_q, mu_p, var_p):
    """
    KL[q || p] for diagonal Gaussians.
    """
    var_q = torch.exp(logvar_q)
    mu_p = mu_p.unsqueeze(0)
    var_p = var_p.unsqueeze(0)

    kl = 0.5 * torch.sum(
        torch.log(var_p) - logvar_q +
        (var_q + (mu_q - mu_p)**2) / var_p - 1.0,
        dim=-1
    )
    return torch.mean(kl)

def variational_loss(y, signal_model, posterior_net, prior_mu, prior_var, sigma):
    """
    ELBO-style loss:
        E_q[-log p(y|theta)] + KL[q(theta|y) || p(theta)]
    """
    mu_q, logvar_q = posterior_net(y)
    theta_sample = posterior_net.reparameterize(mu_q, logvar_q)

    y_pred = signal_model(theta_sample)
    nll = gaussian_neg_log_likelihood(y, y_pred, sigma)
    kl = kl_gaussian(mu_q, logvar_q, prior_mu, prior_var)

    total = nll + kl
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_nll": float(nll.detach().cpu()),
        "loss_kl": float(kl.detach().cpu())
    }
    return total, stats

# ---------------------------------------------------
# Posterior sampling via Langevin dynamics
# ---------------------------------------------------

class LangevinPosteriorSampler:
    """
    Approximate posterior sampling:
        theta_{k+1} = theta_k - eta * grad E(theta_k)
                      + sqrt(2 eta tau) * noise
    """
    def __init__(self, signal_model, prior, sigma=0.01,
                 step_size=1e-3, temperature=1.0, num_steps=500):
        self.signal_model = signal_model
        self.prior = prior
        self.sigma = sigma
        self.step_size = step_size
        self.temperature = temperature
        self.num_steps = num_steps

    def sample(self, y, theta_init):
        theta = theta_init.clone().detach()
        samples = []

        for _ in range(self.num_steps):
            theta.requires_grad_(True)

            energy = posterior_energy(
                theta, y, self.signal_model, self.prior, self.sigma
            )
            grad = torch.autograd.grad(energy, theta)[0]

            noise = torch.randn_like(theta) * torch.sqrt(
                torch.tensor(
                    2.0 * self.step_size * self.temperature,
                    device=theta.device
                )
            )

            theta = (theta - self.step_size * grad + noise).detach()
            samples.append(theta)

        return samples

# ------------------------------------------------------
# Example prior training / variational inference loop
# ------------------------------------------------------

def train_variational_one_epoch(
    posterior_net, signal_model,
    prior_mu, prior_var,
    loader, optimizer, device, sigma=0.01
):
    posterior_net.train()
    running = {}

    for batch in loader:
        y = batch["signal"].to(device)   # measured MRI signal

        optimizer.zero_grad()

        loss, stats = variational_loss(
            y, signal_model, posterior_net,
            prior_mu, prior_var, sigma
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
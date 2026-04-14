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
    Adjoint operator for reconstruction / gradient calculations.
    """
    if smaps is None:
        return ifft2c(mask * y)
    else:
        x_coils = ifft2c(mask.unsqueeze(1) * y)
        return torch.sum(x_coils * torch.conj(smaps), dim=1)

# ------------------------------------------------------------
# Generative prior: simple VAE-style decoder
# ------------------------------------------------------------

class LatentEncoder(nn.Module):
    def __init__(self, in_ch, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.mu(h), self.logvar(h)

class LatentDecoder(nn.Module):
    def __init__(self, latent_dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, H * W * 2)
        )

    def forward(self, z):
        out = self.net(z)
        out = out.view(z.shape[0], self.H, self.W, 2)
        re = out[..., 0]
        im = out[..., 1]
        return torch.complex(re, im)

class GenerativePrior(nn.Module):
    """
    VAE-like generative prior.
    """
    def __init__(self, in_ch, latent_dim, H, W):
        super().__init__()
        self.encoder = LatentEncoder(in_ch, latent_dim)
        self.decoder = LatentDecoder(latent_dim, H, W)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_2ch):
        mu, logvar = self.encoder(x_2ch)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        return x_rec, mu, logvar, z

# ------------------------------------------------------------
# Generative prior training loss
# ------------------------------------------------------------

def vae_loss(x_rec, x_true, mu, logvar, beta=1e-3):
    """
    x_rec, x_true : complex images
    """
    rec_loss = F.mse_loss(x_rec.real, x_true.real) + \
               F.mse_loss(x_rec.imag, x_true.imag)

    kl = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total = rec_loss + beta * kl
    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_rec": float(rec_loss.detach().cpu()),
        "loss_kl": float(kl.detach().cpu())
    }
    return total, stats

# ------------------------------------------------------------
# Physics-constrained latent inference
# ------------------------------------------------------------

class LatentInferenceSolver:
    """
    Optimize latent variable z for a fixed trained decoder G_theta.
    Solves:
        min_z  (1/2sigma^2)||A(G(z)) - y||^2 + lambda_z ||z||^2
    """
    def __init__(self, decoder, sigma=0.01, lambda_z=1e-3,
                 lr=1e-2, num_steps=100):
        self.decoder = decoder
        self.sigma = sigma
        self.lambda_z = lambda_z
        self.lr = lr
        self.num_steps = num_steps

    def solve(self, y, mask, smaps=None, z_init=None):
        """
        y     : measured k-space
        mask  : sampling mask
        """
        B = y.shape[0]

        if z_init is None:
            latent_dim = self.decoder.net[0].in_features
            z = torch.zeros(B, latent_dim, device=y.device,
                            requires_grad=True)
        else:
            z = z_init.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([z], lr=self.lr)

        history = []
        for _ in range(self.num_steps):
            optimizer.zero_grad()

            x = self.decoder(z)
            y_pred = forward_op(x, mask, smaps)

            data_term = torch.mean(torch.abs(y_pred - y)**2) / \
                        (2.0 * self.sigma**2)

            prior_term = self.lambda_z * torch.mean(z**2)

            loss = data_term + prior_term
            loss.backward()
            optimizer.step()

            history.append(float(loss.detach().cpu()))

        x_map = self.decoder(z.detach())
        return x_map, z.detach(), history

# ------------------------------------------------------------
# Optional posterior sampling via Langevin dynamics
# ------------------------------------------------------------

class LatentPosteriorSampler:
    """
    Approximate posterior sampling in latent space:
        z_{k+1} = z_k - eta * grad E(z_k) + sqrt(2 eta tau) * noise
    where
        E(z) = (1/2sigma^2)||A(G(z)) - y||^2 + lambda_z ||z||^2
    """
    def __init__(self, decoder, sigma=0.01, lambda_z=1e-3,
                 step_size=1e-3, temperature=1.0, num_steps=200):
        self.decoder = decoder
        self.sigma = sigma
        self.lambda_z = lambda_z
        self.step_size = step_size
        self.temperature = temperature
        self.num_steps = num_steps

    def sample(self, y, mask, smaps=None, z_init=None):
        B = y.shape[0]
        latent_dim = self.decoder.net[0].in_features

        if z_init is None:
            z = torch.randn(B, latent_dim, device=y.device)
        else:
            z = z_init.clone().detach()

        samples = []

        for _ in range(self.num_steps):
            z.requires_grad_(True)

            x = self.decoder(z)
            y_pred = forward_op(x, mask, smaps)

            data_term = torch.mean(torch.abs(y_pred - y)**2) / \
                        (2.0 * self.sigma**2)
            prior_term = self.lambda_z * torch.mean(z**2)
            energy = data_term + prior_term

            grad = torch.autograd.grad(energy, z)[0]

            noise = torch.randn_like(z) * \
                    torch.sqrt(torch.tensor(
                        2.0 * self.step_size * self.temperature,
                        device=z.device
                    ))

            z = (z - self.step_size * grad + noise).detach()

            samples.append(self.decoder(z).detach())

        return samples

# ------------------------------------------------------------
# Unified model wrapper
# ------------------------------------------------------------

class UnifiedPhysicsGenerativeMRI(nn.Module):
    """
    Wrapper around the generative prior and inference model.
    """
    def __init__(self, in_ch, latent_dim, H, W):
        super().__init__()
        self.prior = GenerativePrior(in_ch, latent_dim, H, W)

    def train_prior_step(self, x_true, optimizer, beta=1e-3):
        """
        Train the generative prior on fully sampled images.
        x_true : (B, H, W) complex
        """
        optimizer.zero_grad()

        x_2ch = torch.stack([x_true.real, x_true.imag], dim=1)
        x_rec, mu, logvar, z = self.prior(x_2ch)

        loss, stats = vae_loss(x_rec, x_true, mu, logvar, beta=beta)
        loss.backward()
        optimizer.step()

        return stats

    def infer_map(self, y, mask, smaps=None,
                  sigma=0.01, lambda_z=1e-3,
                  lr=1e-2, num_steps=100):
        solver = LatentInferenceSolver(
            self.prior.decoder,
            sigma=sigma,
            lambda_z=lambda_z,
            lr=lr,
            num_steps=num_steps
        )
        return solver.solve(y, mask, smaps)

    def sample_posterior(self, y, mask, smaps=None,
                         sigma=0.01, lambda_z=1e-3,
                         step_size=1e-3, temperature=1.0,
                         num_steps=200):
        sampler = LatentPosteriorSampler(
            self.prior.decoder,
            sigma=sigma,
            lambda_z=lambda_z,
            step_size=step_size,
            temperature=temperature,
            num_steps=num_steps
        )
        return sampler.sample(y, mask, smaps)

# ------------------------------------------------------------
# Example prior training loop
# ------------------------------------------------------------

def train_prior_one_epoch(model, loader, optimizer, device, beta=1e-3):
    model.train()
    running = {}

    for batch in loader:
        x_true = batch["image"].to(device)   # fully sampled complex image
        stats = model.train_prior_step(x_true, optimizer, beta=beta)

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
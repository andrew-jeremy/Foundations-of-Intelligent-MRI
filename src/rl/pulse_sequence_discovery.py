import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ------------------------------------------------------------
# Bloch dynamics utilities
# ------------------------------------------------------------

def rotation_matrix_x(alpha):
    c = torch.cos(alpha)
    s = torch.sin(alpha)
    R = torch.zeros(alpha.shape[0], 3, 3, device=alpha.device)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c
    return R

def rotation_matrix_z(phi):
    c = torch.cos(phi)
    s = torch.sin(phi)
    R = torch.zeros(phi.shape[0], 3, 3, device=phi.device)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R

def apply_rf_pulse(M, alpha, phase):
    """
    Apply an RF pulse with flip angle alpha and phase phi.
    A simple implementation rotates around x after phase alignment.
    M : (B, 3)
    alpha, phase : (B,)
    """
    Rz_neg = rotation_matrix_z(-phase)
    Rx = rotation_matrix_x(alpha)
    Rz_pos = rotation_matrix_z(phase)

    M1 = torch.einsum('bij,bj->bi', Rz_neg, M)
    M2 = torch.einsum('bij,bj->bi', Rx, M1)
    M3 = torch.einsum('bij,bj->bi', Rz_pos, M2)
    return M3

def free_precession_relaxation(M, dt, T1, T2, off_res, M0=1.0):
    """
    Approximate Bloch evolution under relaxation and off-resonance.
    M : (B, 3)
    dt, T1, T2, off_res : (B,)
    """
    E1 = torch.exp(-dt / T1)
    E2 = torch.exp(-dt / T2)

    phi = off_res * dt
    c = torch.cos(phi)
    s = torch.sin(phi)

    Mx = E2 * (c * M[:, 0] - s * M[:, 1])
    My = E2 * (s * M[:, 0] + c * M[:, 1])
    Mz = M0 - (M0 - M[:, 2]) * E1

    return torch.stack([Mx, My, Mz], dim=-1)

def transverse_signal(M):
    """
    Complex transverse MRI signal.
    """
    return torch.complex(M[:, 0], M[:, 1])

# ------------------------------------------------------------
# MRI pulse-sequence discovery environment
# ------------------------------------------------------------

class PulseSequenceEnv:
    """
    RL environment for discovering pulse sequences.
    State:
      current magnetization + optional latent tissue parameters +
      current time budget / SAR budget.
    Action:
      RF flip angle, RF phase, waiting time, optional gradient proxy.
    """
    def __init__(self, tissue_params, max_steps=20, M0=1.0):
        """
        tissue_params: dict with tensors
          T1      : (B,)
          T2      : (B,)
          off_res : (B,)
        """
        self.T1 = tissue_params["T1"]
        self.T2 = tissue_params["T2"]
        self.off_res = tissue_params["off_res"]
        self.batch_size = self.T1.shape[0]

        self.max_steps = max_steps
        self.M0 = M0

        self.reset()

    def reset(self):
        self.step_id = 0

        # start at thermal equilibrium
        self.M = torch.zeros(self.batch_size, 3, device=self.T1.device)
        self.M[:, 2] = self.M0

        self.time_used = torch.zeros(self.batch_size, device=self.T1.device)
        self.sar_used = torch.zeros(self.batch_size, device=self.T1.device)

        # simple scalar uncertainty proxy, initialized high
        self.uncertainty = torch.ones(self.batch_size, device=self.T1.device)

        return self.get_state()

    def get_state(self):
        """
        State vector:
          [Mx, My, Mz, T1, T2, off_res, time_used, sar_used, uncertainty]
        In practice, one may remove latent parameters if hidden.
        """
        return torch.cat([
            self.M,
            self.T1.unsqueeze(-1),
            self.T2.unsqueeze(-1),
            self.off_res.unsqueeze(-1),
            self.time_used.unsqueeze(-1),
            self.sar_used.unsqueeze(-1),
            self.uncertainty.unsqueeze(-1)
        ], dim=-1)

    def step(self, action):
        """
        action: dict with tensors
          alpha : (B,) RF flip angle in radians
          phase : (B,) RF phase
          dt    : (B,) time interval
          grad  : (B,) optional gradient proxy
        """
        alpha = action["alpha"]
        phase = action["phase"]
        dt = action["dt"]
        grad = action["grad"]

        prev_uncertainty = self.uncertainty.clone()

        # Apply RF pulse
        self.M = apply_rf_pulse(self.M, alpha, phase)

        # Free precession / relaxation
        self.M = free_precession_relaxation(
            self.M, dt, self.T1, self.T2, self.off_res, M0=self.M0
        )

        # Measurement
        y = transverse_signal(self.M)

        # Update simple budgets
        self.time_used = self.time_used + dt
        self.sar_used = self.sar_used + alpha**2

        # Surrogate uncertainty update:
        # higher transverse signal magnitude -> lower uncertainty
        signal_power = torch.abs(y)
        self.uncertainty = self.uncertainty / (1.0 + signal_power)

        # Reward: uncertainty reduction minus cost penalties
        reward = (
            (prev_uncertainty - self.uncertainty)
            - 1e-3 * alpha**2
            - 1e-3 * dt
            - 1e-4 * grad**2
        )

        self.step_id += 1
        done = (self.step_id >= self.max_steps)

        next_state = self.get_state()
        info = {
            "signal": y,
            "uncertainty": self.uncertainty
        }
        return next_state, reward, done, info

# ------------------------------------------------------------
# Policy / actor network
# ------------------------------------------------------------

class PulsePolicyNet(nn.Module):
    """
    Actor network outputs mean and std for continuous pulse parameters.
    """
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.alpha_mu = nn.Linear(hidden_dim, 1)
        self.phase_mu = nn.Linear(hidden_dim, 1)
        self.dt_mu    = nn.Linear(hidden_dim, 1)
        self.grad_mu  = nn.Linear(hidden_dim, 1)

        self.log_std = nn.Parameter(torch.zeros(4))

    def forward(self, state):
        h = self.net(state)

        alpha_mu = self.alpha_mu(h)
        phase_mu = self.phase_mu(h)
        dt_mu    = self.dt_mu(h)
        grad_mu  = self.grad_mu(h)

        mu = torch.cat([alpha_mu, phase_mu, dt_mu, grad_mu], dim=-1)
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        return mu, std

    def sample_action(self, state):
        mu, std = self(state)
        dist = Normal(mu, std)
        raw = dist.rsample()   # reparameterized sample

        # enforce practical ranges
        alpha = torch.sigmoid(raw[:, 0]) * torch.pi          # [0, pi]
        phase = torch.tanh(raw[:, 1]) * torch.pi             # [-pi, pi]
        dt    = F.softplus(raw[:, 2]) + 1e-4                 # > 0
        grad  = torch.tanh(raw[:, 3])                        # normalized

        # log prob of raw Gaussian sample
        log_prob = dist.log_prob(raw).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        action = {
            "alpha": alpha,
            "phase": phase,
            "dt": dt,
            "grad": grad
        }
        return action, log_prob, entropy

# ------------------------------------------------------------
# Value / critic network
# ------------------------------------------------------------

class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.net(state).squeeze(-1)

# ------------------------------------------------------------
# Rollout collection
# ------------------------------------------------------------

def collect_episode(env, actor, critic):
    states = []
    log_probs = []
    entropies = []
    rewards = []
    values = []

    state = env.reset()
    done = False

    while not done:
        value = critic(state)
        action, log_prob, entropy = actor.sample_action(state)

        next_state, reward, done, info = env.step(action)

        states.append(state)
        log_probs.append(log_prob)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(value)

        state = next_state

    values.append(critic(state))  # bootstrap value for terminal state
    return states, log_probs, entropies, rewards, values

# ------------------------------------------------------------
# Returns and advantages
# ------------------------------------------------------------

def compute_returns_advantages(rewards, values, gamma=0.99, lam=0.95):
    """
    Generalized advantage estimation.
    rewards: list of tensors (B,)
    values : list of tensors (B,), length = len(rewards)+1
    """
    T = len(rewards)
    advantages = []
    gae = 0.0

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]

    advantages = torch.stack(advantages, dim=0)
    returns = torch.stack(returns, dim=0)

    # normalize over time and batch
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages

# ------------------------------------------------------------
# Actor-critic training step
# ------------------------------------------------------------

def actor_critic_step(env, actor, critic, actor_opt, critic_opt,
                      gamma=0.99, lam=0.95, entropy_weight=1e-3):
    states, log_probs, entropies, rewards, values = collect_episode(
        env, actor, critic
    )

    returns, advantages = compute_returns_advantages(
        rewards, values, gamma=gamma, lam=lam
    )

    log_probs = torch.stack(log_probs, dim=0)
    entropies = torch.stack(entropies, dim=0)
    values_t = torch.stack(values[:-1], dim=0)

    actor_loss = -(log_probs * advantages.detach()).mean() \
                 - entropy_weight * entropies.mean()

    critic_loss = F.mse_loss(values_t, returns.detach())

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    total_reward = torch.stack(rewards, dim=0).sum(dim=0).mean()

    stats = {
        "actor_loss": float(actor_loss.detach().cpu()),
        "critic_loss": float(critic_loss.detach().cpu()),
        "episode_reward": float(total_reward.detach().cpu())
    }
    return stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_pulse_discovery(env, actor, critic, actor_opt, critic_opt,
                          num_episodes=1000):
    history = []

    for ep in range(num_episodes):
        stats = actor_critic_step(
            env, actor, critic,
            actor_opt, critic_opt,
            gamma=0.99, lam=0.95, entropy_weight=1e-3
        )
        history.append(stats)

    return history
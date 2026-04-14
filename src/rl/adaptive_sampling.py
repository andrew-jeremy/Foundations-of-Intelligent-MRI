import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ------------------------------------------------------------
# Basic MRI utilities
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

def zero_filled_recon(kspace, mask):
    return ifft2c(kspace * mask)

# ------------------------------------------------------------
# MRI adaptive sampling environment
# ------------------------------------------------------------

class AdaptiveMRIEnv:
    """
    Simplified environment for adaptive k-space sampling.

    State contains:
      - current zero-filled reconstruction
      - current sampling mask
      - current time/frame index (optional)

    Action:
      - choose one candidate k-space line not yet sampled
    """
    def __init__(self, x_true_seq, candidate_masks, max_steps):
        """
        x_true_seq       : (T, H, W) complex dynamic image sequence
        candidate_masks  : (K, H, W) binary masks for candidate actions
        max_steps        : number of RL sampling decisions
        """
        self.x_true_seq = x_true_seq
        self.k_true_seq = fft2c(x_true_seq)   # ground-truth k-space
        self.candidate_masks = candidate_masks
        self.K = candidate_masks.shape[0]
        self.max_steps = max_steps
        self.T, self.H, self.W = x_true_seq.shape

        self.reset()

    def reset(self):
        self.step_id = 0
        self.frame_id = 0

        # start with empty sampling mask
        self.mask = torch.zeros(self.H, self.W, dtype=torch.float32)
        self.k_obs = torch.zeros(
            self.H, self.W, dtype=self.k_true_seq.dtype
        )

        self.x_hat = zero_filled_recon(self.k_obs, self.mask)
        return self.get_state()

    def get_state(self):
        """
        Returns a state tensor for the policy network.
        Here we use:
          real(x_hat), imag(x_hat), mask
        -> shape (3, H, W)
        """
        return torch.stack([
            self.x_hat.real.float(),
            self.x_hat.imag.float(),
            self.mask.float()
        ], dim=0)

    def available_actions(self):
        """
        Mask out candidate actions already fully sampled.
        """
        avail = []
        for k in range(self.K):
            cand = self.candidate_masks[k]
            # available if it adds at least one new point
            new_points = torch.sum(cand * (1.0 - self.mask))
            avail.append(new_points > 0)
        return torch.tensor(avail, dtype=torch.bool)

    def step(self, action):
        """
        Apply one sampling action.
        """
        prev_error = torch.mean(
            torch.abs(self.x_true_seq[self.frame_id] - self.x_hat)**2
        )

        # update mask
        cand = self.candidate_masks[action]
        self.mask = torch.clamp(self.mask + cand, max=1.0)

        # acquire measured samples from true k-space
        k_true = self.k_true_seq[self.frame_id]
        self.k_obs = self.mask * k_true

        # update reconstruction
        self.x_hat = zero_filled_recon(self.k_obs, self.mask)

        new_error = torch.mean(
            torch.abs(self.x_true_seq[self.frame_id] - self.x_hat)**2
        )

        # reward = improvement in reconstruction
        reward = (prev_error - new_error).real.float()

        self.step_id += 1
        done = (self.step_id >= self.max_steps)

        next_state = self.get_state()
        info = {
            "prev_error": prev_error.item(),
            "new_error": new_error.item()
        }

        return next_state, reward, done, info

# ------------------------------------------------------------
# Policy network
# ------------------------------------------------------------

class SamplingPolicyNet(nn.Module):
    """
    CNN policy for adaptive k-space sampling.
    Input:
      state tensor (B, 3, H, W)
    Output:
      logits over K candidate sampling actions
    """
    def __init__(self, H, W, K):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, K)
        )

    def forward(self, state):
        z = self.encoder(state)
        logits = self.head(z)
        return logits

# ------------------------------------------------------------
# Policy-gradient rollout
# ------------------------------------------------------------

def collect_episode(env, policy, device):
    """
    Roll out one episode using current policy.
    """
    log_probs = []
    rewards = []
    entropies = []

    state = env.reset().unsqueeze(0).to(device)  # (1,3,H,W)

    done = False
    while not done:
        logits = policy(state)   # (1,K)

        # mask unavailable actions
        avail = env.available_actions().to(device).unsqueeze(0)
        logits = logits.masked_fill(~avail, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        next_state, reward, done, info = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(torch.tensor([reward], device=device))
        entropies.append(entropy)

        state = next_state.unsqueeze(0).to(device)

    return log_probs, rewards, entropies

# ------------------------------------------------------------
# Return computation
# ------------------------------------------------------------

def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.cat(returns, dim=0)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns

# ------------------------------------------------------------
# REINFORCE training step
# ------------------------------------------------------------

def reinforce_step(env, policy, optimizer, device,
                   gamma=0.99, entropy_weight=1e-3):
    policy.train()
    optimizer.zero_grad()

    log_probs, rewards, entropies = collect_episode(env, policy, device)
    returns = compute_returns(rewards, gamma=gamma)

    policy_loss = 0.0
    entropy_bonus = 0.0

    for log_prob, G, entropy in zip(log_probs, returns, entropies):
        policy_loss = policy_loss - log_prob * G
        entropy_bonus = entropy_bonus + entropy

    loss = policy_loss - entropy_weight * entropy_bonus
    loss.backward()
    optimizer.step()

    total_reward = sum([r.item() for r in rewards])
    stats = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "entropy_bonus": float(entropy_bonus.detach().cpu()),
        "episode_reward": total_reward
    }
    return stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_policy(env, policy, optimizer, device, num_episodes):
    history = []

    for ep in range(num_episodes):
        stats = reinforce_step(
            env, policy, optimizer, device,
            gamma=0.99,
            entropy_weight=1e-3
        )
        history.append(stats)

    return history
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# Patch embedding
# ------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Convert image into patch tokens.
    """
    def __init__(self, in_ch=1, embed_dim=128, patch_size=8):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_ch, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x : (B, C, H, W)
        returns:
            tokens : (B, N, D)
            H_p, W_p : patch grid size
        """
        x = self.proj(x)                # (B, D, H_p, W_p)
        B, D, H_p, W_p = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, H_p, W_p

# ------------------------------------------------------------
# Positional encoding
# ------------------------------------------------------------

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_tokens, embed_dim):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, max_tokens, embed_dim))

    def forward(self, x):
        """
        x : (B, N, D)
        """
        N = x.shape[1]
        return x + self.pos[:, :N, :]

# ------------------------------------------------------------
# Transformer block
# ------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        """
        x : (B, N, D)
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------------------------------------
# Cross-attention block for reference/moving interaction
# ------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x_ref, x_mov):
        """
        x_ref : (B, N, D)
        x_mov : (B, N, D)
        Use reference queries and moving keys/values.
        """
        q = self.norm_q(x_ref)
        kv = self.norm_kv(x_mov)
        attn_out, _ = self.attn(q, kv, kv)
        x = x_ref + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# ------------------------------------------------------------
# Motion decoder
# ------------------------------------------------------------

class MotionDecoder(nn.Module):
    """
    Decode transformer tokens into dense displacement field.
    """
    def __init__(self, embed_dim=128, patch_size=8, out_ch=2):
        super().__init__()
        self.patch_size = patch_size
        self.out_ch = out_ch

        self.conv1 = nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3, padding=1)

    def forward(self, tokens, H_p, W_p, H, W):
        """
        tokens : (B, N, D)
        returns dense flow field : (B, 2, H, W)
        """
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, D, H_p, W_p)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)

        # Upsample patch-grid flow to full resolution
        flow = F.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True
        )
        return flow

# ------------------------------------------------------------
# Spatial warping
# ------------------------------------------------------------

def warp_image(x, flow):
    """
    x    : (B, C, H, W)
    flow : (B, 2, H, W), displacement in pixels
    """
    B, C, H, W = x.shape
    device = x.device

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    base_grid = torch.stack([xx, yy], dim=-1)      # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(B, 1, 1, 1)

    # normalize flow to [-1,1] coordinates
    flow_x = flow[:, 0] / max(W - 1, 1) * 2.0
    flow_y = flow[:, 1] / max(H - 1, 1) * 2.0
    flow_grid = torch.stack([flow_x, flow_y], dim=-1)

    sampling_grid = base_grid + flow_grid
    x_warp = F.grid_sample(
        x, sampling_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    return x_warp

# ------------------------------------------------------------
# Full transformer-based motion model
# ------------------------------------------------------------

class TransformerMotionModel(nn.Module):
    """
    Estimate motion field between reference and moving image.
    """
    def __init__(
        self,
        in_ch=1,
        embed_dim=128,
        patch_size=8,
        num_blocks=4,
        num_heads=4
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_ch=in_ch,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
        self.pos_enc = LearnablePositionalEncoding(
            max_tokens=4096,
            embed_dim=embed_dim
        )

        self.ref_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])
        self.mov_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(num_blocks)
        ])

        self.cross_block = CrossAttentionBlock(embed_dim, num_heads)
        self.decoder = MotionDecoder(embed_dim, patch_size, out_ch=2)

    def forward(self, x_ref, x_mov):
        """
        x_ref, x_mov : (B, 1, H, W)
        returns:
            flow      : (B, 2, H, W)
            x_warped  : (B, 1, H, W)
        """
        B, C, H, W = x_ref.shape

        tok_ref, H_p, W_p = self.patch_embed(x_ref)
        tok_mov, _, _ = self.patch_embed(x_mov)

        tok_ref = self.pos_enc(tok_ref)
        tok_mov = self.pos_enc(tok_mov)

        for blk in self.ref_blocks:
            tok_ref = blk(tok_ref)

        for blk in self.mov_blocks:
            tok_mov = blk(tok_mov)

        fused = self.cross_block(tok_ref, tok_mov)
        flow = self.decoder(fused, H_p, W_p, H, W)
        x_warped = warp_image(x_mov, flow)

        return flow, x_warped

# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------

def image_similarity_loss(x_ref, x_warped):
    """
    Simple MSE similarity loss.
    Could be replaced with NCC, SSIM, mutual information, etc.
    """
    return F.mse_loss(x_warped, x_ref)

def smoothness_loss(flow):
    """
    Spatial smoothness penalty on deformation field.
    """
    dx = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return torch.mean(dx**2) + torch.mean(dy**2)

def supervised_flow_loss(flow_pred, flow_true):
    return F.mse_loss(flow_pred, flow_true)

def motion_model_loss(
    x_ref,
    x_warped,
    flow,
    flow_true=None,
    lambda_smooth=1e-2,
    lambda_sup=1.0
):
    loss_sim = image_similarity_loss(x_ref, x_warped)
    loss_smooth = smoothness_loss(flow)

    loss_sup = 0.0
    if flow_true is not None:
        loss_sup = supervised_flow_loss(flow, flow_true)

    total = loss_sim + lambda_smooth * loss_smooth + lambda_sup * loss_sup

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_similarity": float(loss_sim.detach().cpu()),
        "loss_smooth": float(loss_smooth.detach().cpu()),
        "loss_supervised": float(loss_sup.detach().cpu())
            if not isinstance(loss_sup, float) else loss_sup
    }
    return total, stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_motion_model_one_epoch(model, loader, optimizer, device):
    model.train()
    running = {}

    for batch in loader:
        x_ref = batch["x_ref"].to(device)   # (B,1,H,W)
        x_mov = batch["x_mov"].to(device)   # (B,1,H,W)
        flow_true = batch.get("flow_true", None)

        if flow_true is not None:
            flow_true = flow_true.to(device)

        optimizer.zero_grad()

        flow, x_warped = model(x_ref, x_mov)

        loss, stats = motion_model_loss(
            x_ref=x_ref,
            x_warped=x_warped,
            flow=flow,
            flow_true=flow_true,
            lambda_smooth=1e-2,
            lambda_sup=1.0 if flow_true is not None else 0.0
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running

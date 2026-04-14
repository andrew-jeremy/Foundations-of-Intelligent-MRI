import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------
# Graph utilities
# ---------------------------------------------

def normalize_adjacency(A):
    """
    A : (N, N) adjacency matrix
    returns normalized adjacency:
        A_tilde = D^{-1/2} (A + I) D^{-1/2}
    """
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)
    A_hat = A + I
    deg = torch.sum(A_hat, dim=1)
    deg_inv_sqrt = torch.pow(deg + 1e-12, -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt

# ------------------------------------------------
# Basic graph convolution layer
# ------------------------------------------------

class GraphConv(nn.Module):
    """
    Simple graph convolution:
        H' = sigma(A_tilde H W)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, A_tilde):
        """
        x       : (B, N, F_in)
        A_tilde : (N, N)
        """
        xw = self.lin(x)                     # (B, N, F_out)
        out = torch.einsum('nm,bmf->bnf', A_tilde, xw)
        return F.relu(out)

# -------------------------------------------------
# Spatiotemporal GNN block
# -------------------------------------------------

class STGNNBlock(nn.Module):
    """
    One block of:
      spatial graph convolution
      temporal recurrent modeling
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.graph_conv = GraphConv(in_features, hidden_features)
        self.gru = nn.GRU(
            input_size=hidden_features,
            hidden_size=hidden_features,
            batch_first=True
        )

    def forward(self, x_seq, A_tilde):
        """
        x_seq   : (B, T, N, F_in)
        returns : (B, T, N, F_hidden)
        """
        B, T, N, F_in = x_seq.shape

        # spatial graph convolution at each time point
        spatial_out = []
        for t in range(T):
            xt = x_seq[:, t]                          # (B, N, F_in)
            ht = self.graph_conv(xt, A_tilde)        # (B, N, F_hidden)
            spatial_out.append(ht)

        h_spatial = torch.stack(spatial_out, dim=1)  # (B, T, N, F_hidden)

        # temporal modeling nodewise
        node_outputs = []
        for n in range(N):
            hn = h_spatial[:, :, n, :]               # (B, T, F_hidden)
            hn_out, _ = self.gru(hn)                 # (B, T, F_hidden)
            node_outputs.append(hn_out)

        h_temporal = torch.stack(node_outputs, dim=2)  # (B, T, N, F_hidden)
        return h_temporal

# --------------------------------------------------
# Full spatiotemporal graph neural network
# --------------------------------------------------

class SpatiotemporalGNN(nn.Module):
    """
    End-to-end spatiotemporal GNN.
    Example tasks:
      - sequence denoising
      - nodewise prediction
      - graph-level classification
      - future-state forecasting
    """
    def __init__(
        self,
        in_features,
        hidden_features,
        out_features,
        num_blocks=2,
        graph_level_task=False
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        self.blocks.append(STGNNBlock(in_features, hidden_features))
        for _ in range(num_blocks - 1):
            self.blocks.append(STGNNBlock(hidden_features, hidden_features))

        self.graph_level_task = graph_level_task

        if graph_level_task:
            self.readout = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )
        else:
            self.readout = nn.Sequential(
                nn.Linear(hidden_features, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, out_features)
            )

    def forward(self, x_seq, A):
        """
        x_seq : (B, T, N, F_in)
        A     : (N, N)
        """
        A_tilde = normalize_adjacency(A)

        h = x_seq
        for block in self.blocks:
            h = block(h, A_tilde)   # (B, T, N, hidden)

        if self.graph_level_task:
            # pool over nodes and time
            h_pool = h.mean(dim=2).mean(dim=1)  # (B, hidden)
            out = self.readout(h_pool)          # (B, out_features)
        else:
            # nodewise / timewise prediction
            out = self.readout(h)               # (B, T, N, out_features)

        return out, h

# ------------------------------------------------------------
# Optional temporal attention module
# ------------------------------------------------------------

class TemporalSelfAttention(nn.Module):
    """
    Replace or augment GRU with temporal self-attention.
    """
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x):
        """
        x : (B, T, hidden_dim)
        """
        out, _ = self.attn(x, x, x)
        return out

# ------------------------------------------------------------
# Example loss functions
# ------------------------------------------------------------

def nodewise_regression_loss(pred, target, mask=None):
    """
    pred, target : (B, T, N, F_out)
    """
    err = (pred - target)**2
    if mask is not None:
        err = err * mask
    return err.mean()

def graph_classification_loss(logits, labels):
    return F.cross_entropy(logits, labels)

def temporal_smoothness_loss(pred):
    """
    Encourages smooth temporal trajectories.
    pred : (B, T, N, F_out)
    """
    return ((pred[:, 1:] - pred[:, :-1])**2).mean()

def graph_laplacian_loss(h, A):
    """
    Spatial smoothness over the graph using Laplacian regularization.
    h : (B, T, N, Hdim)
    """
    N = A.shape[0]
    D = torch.diag(torch.sum(A, dim=1))
    L = D - A
    loss = 0.0

    for b in range(h.shape[0]):
        for t in range(h.shape[1]):
            Hbt = h[b, t]  # (N, Hdim)
            loss = loss + torch.trace(Hbt.T @ L @ Hbt)

    return loss / (h.shape[0] * h.shape[1])

# ------------------------------------------------------------
# Full objective
# ------------------------------------------------------------

def stgnn_loss(
    pred, hidden, target, A,
    labels=None,
    task_type="regression",
    lambda_temp=1e-3,
    lambda_graph=1e-3
):
    if task_type == "regression":
        loss_main = nodewise_regression_loss(pred, target)
    elif task_type == "classification":
        loss_main = graph_classification_loss(pred, labels)
    else:
        raise ValueError("Unknown task_type")

    loss_temp = 0.0
    if task_type == "regression":
        loss_temp = temporal_smoothness_loss(pred)

    loss_graph = graph_laplacian_loss(hidden, A)

    total = (
        loss_main +
        lambda_temp * loss_temp +
        lambda_graph * loss_graph
    )

    stats = {
        "loss_total": float(total.detach().cpu()),
        "loss_main": float(loss_main.detach().cpu()),
        "loss_temp": float(loss_temp.detach().cpu())
            if not isinstance(loss_temp, float) else loss_temp,
        "loss_graph": float(loss_graph.detach().cpu())
    }

    return total, stats

# ------------------------------------------------------------
# Example training loop
# ------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, task_type="regression"):
    model.train()
    running = {}

    for batch in loader:
        x_seq = batch["x_seq"].to(device)   # (B, T, N, F_in)
        A = batch["adjacency"].to(device)   # (N, N)

        target = batch.get("target", None)
        labels = batch.get("labels", None)

        if target is not None:
            target = target.to(device)
        if labels is not None:
            labels = labels.to(device)

        optimizer.zero_grad()

        pred, hidden = model(x_seq, A)

        loss, stats = stgnn_loss(
            pred, hidden, target, A,
            labels=labels,
            task_type=task_type,
            lambda_temp=1e-3,
            lambda_graph=1e-3
        )

        loss.backward()
        optimizer.step()

        for k, v in stats.items():
            running[k] = running.get(k, 0.0) + v

    for k in running:
        running[k] /= len(loader)

    return running
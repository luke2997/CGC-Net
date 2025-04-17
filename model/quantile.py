import math
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import DenseSAGEConv, DenseGINConv


# ----------------------------------------------------------------
# 1.  Quantile neighbourhood aggregator
# ----------------------------------------------------------------
def quantile_aggregate(x, adj, *, quantiles=(0.25, 0.5, 0.75), weights=None):
    """
    Aggregate neighbour features with a weighted combination of quantiles.

    Parameters
    ----------
    x         : FloatTensor [B, N, C]   – node features for a *dense* batch
    adj       : FloatTensor [B, N, N]   – 0/1 (or soft) adjacency
    quantiles : tuple[float]            – e.g. (0.1, 0.5, 0.9)
    weights   : 1‑D tensor | None       – same length as `quantiles`

    Returns
    -------
    out       : FloatTensor [B, N, C]   – aggregated representation
    """
    B, N, C = x.shape
    device = x.device

    # Guarantee at least one neighbour (itself)
    idx = torch.arange(N, device=device)
    adj = adj.clone()
    adj[:, idx, idx] = 1.0

    # Broadcast features to every target node, then NaN‑mask non‑neighbours
    x_exp   = x.unsqueeze(1).expand(B, N, N, C)          # [B,N,N,C]
    mask    = adj.unsqueeze(-1)                          # [B,N,N,1]
    x_mask  = torch.where(mask > 0, x_exp, torch.full_like(x_exp, float('nan')))

    # Compute all required quantiles in one pass: [M,B,N,C]
    taus    = torch.tensor(quantiles, device=device, dtype=x.dtype)
    qs      = torch.nanquantile(x_mask, taus, dim=2)     # [M,B,N,C]
    qs      = qs.permute(1, 2, 0, 3)                     # [B,N,M,C]

    if weights is None:
        weights = torch.full((len(quantiles),), 1.0 / len(quantiles), device=device, dtype=x.dtype)
    weights = weights.view(1, 1, len(quantiles), 1)

    return (qs * weights).sum(dim=2)                     # [B,N,C]


# ----------------------------------------------------------------
# 2.  Convolution layers that *use* the aggregator
# ----------------------------------------------------------------
class QuantileDenseSAGEConv(DenseSAGEConv):
    """
    Dense GraphSAGE with quantile aggregation instead of the mean.
    All other behaviour mirrors `DenseSAGEConv` (self‑loops, normalisation).
    """
    def __init__(self, in_channels, out_channels,
                 quantiles=(0.25, 0.5, 0.75), q_weights=None,
                 bias=True, normalize=True, **kwargs):
        super().__init__(in_channels, out_channels,
                         bias=bias, normalize=normalize, **kwargs)
        self.quantiles = quantiles
        if q_weights is not None:
            self.register_buffer('q_weights', torch.tensor(q_weights, dtype=torch.float))
        else:
            self.q_weights = None

    def forward(self, x, adj, mask=None, add_loop=False):
        if add_loop:
            idx = torch.arange(adj.size(1), device=adj.device)
            adj = adj.clone()
            adj[:, idx, idx] = 1.0

        if mask is not None:          # mask padded nodes
            adj = adj * mask
            x   = x   * mask

        neigh = quantile_aggregate(x, adj,
                                   quantiles=self.quantiles,
                                   weights=self.q_weights)

        out = self.lin_l(neigh) + self.lin_r(x)
        if self.normalize:
            out = F.normalize(out, p=2.0, dim=-1)
        return out


class QuantileDenseGINConv(DenseGINConv):
    """
    GIN variant with quantile aggregation.
    `nn` is the usual MLP passed to GIN.
    """
    def __init__(self, nn, quantiles=(0.25, 0.5, 0.75), q_weights=None,
                 eps=0., train_eps=False, **kwargs):
        super().__init__(nn, eps=eps, train_eps=train_eps, **kwargs)
        self.quantiles = quantiles
        if q_weights is not None:
            self.register_buffer('q_weights', torch.tensor(q_weights, dtype=torch.float))
        else:
            self.q_weights = None

    def forward(self, x, adj, mask=None, add_loop=False):
        if add_loop:
            idx = torch.arange(adj.size(1), device=adj.device)
            adj = adj.clone()
            adj[:, idx, idx] = 1.0

        if mask is not None:
            adj = adj * mask
            x   = x   * mask

        neigh = quantile_aggregate(x, adj,
                                   quantiles=self.quantiles,
                                   weights=self.q_weights)

        out = self.nn((1 + self.eps) * x + neigh)
        return out

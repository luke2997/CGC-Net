"""
--------------------------------
This code compares:
  1) Standard GCN
  2) Dilated GCN (Linear skipping) 
  3) Dilated GCN (Exponential skipping)
on a ring-of-clusters in:
  - Sparse bridging=1,
  - Dense bridging=25.

We also vary the threshold T in [2,5] to see how it affects neighbour skipping.
Depths tested: [2,4,6,8,10,12,14].
Average similarity also can be added to test for oversmoothing.. 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

###############################################################################
# 1) Generate ring-of-clusters
###############################################################################
def generate_ring_of_clusters(
    num_clusters=5,
    cluster_size=20,
    bridging=1,        # edges to connect each cluster to the next
    feature_dim=8,
    seed=42
):
    rng = np.random.RandomState(seed)
    N = num_clusters * cluster_size
    labels = np.zeros(N, dtype=int)
    for c in range(num_clusters):
        labels[c*cluster_size:(c+1)*cluster_size] = c

    # Adjacency
    A = np.zeros((N, N), dtype=np.float32)
    # Fully connect each cluster
    for c in range(num_clusters):
        start = c*cluster_size
        end   = start+cluster_size
        for i in range(start, end):
            for j in range(i+1, end):
                A[i,j] = 1
                A[j,i] = 1

    # bridging edges in a ring
    for c in range(num_clusters):
        nxt = (c+1) % num_clusters
        c_start, c_end = c*cluster_size, (c+1)*cluster_size
        n_start, n_end = nxt*cluster_size, (nxt+1)*cluster_size
        for _ in range(bridging):
            i = rng.randint(c_start, c_end)
            j = rng.randint(n_start, n_end)
            A[i,j] = 1
            A[j,i] = 1

    # Node features => offset per cluster + random
    X = np.zeros((N, feature_dim), dtype=np.float32)
    for c in range(num_clusters):
        offset = c * 3.0
        start = c*cluster_size
        end   = start+cluster_size
        noise = rng.randn(cluster_size, feature_dim).astype(np.float32)
        X[start:end,:] = offset + noise

    return (torch.from_numpy(A),
            torch.from_numpy(X),
            torch.from_numpy(labels).long())


###############################################################################
# 2) Standard GCN
###############################################################################
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, A):
        deg = A.sum(dim=1, keepdim=True)
        deg[deg==0] = 1e-6
        agg = torch.mm(A, x) / deg
        return self.linear(agg)

class StandardGCN(nn.Module):
    """
    Multi-layer GCN with mean aggregator + ReLU between layers.
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        layers = []
        layers.append(SimpleGCNLayer(in_dim, hidden_dim))
        for _ in range(num_layers - 2):
            layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        layers.append(SimpleGCNLayer(hidden_dim, out_dim))
        self.layers = nn.ModuleList(layers)
        self.act = nn.ReLU()

    def forward(self, x, A):
        for layer in self.layers[:-1]:
            x = layer(x, A)
            x = self.act(x)
        x = self.layers[-1](x, A)
        return x


###############################################################################
# 3) Quantile Aggregation + Dilated Layers
###############################################################################
def quantile_aggregation(neighbor_feats: torch.Tensor,
                         quantiles=[0.25, 0.5, 0.75],
                         weights=None) -> torch.Tensor:
    if neighbor_feats.size(0)==0:
        return torch.zeros(neighbor_feats.size(1), device=neighbor_feats.device)
    if weights is None:
        weights = [1./len(quantiles)]*len(quantiles)
    sorted_feats, _ = torch.sort(neighbor_feats, dim=0)
    n, d = sorted_feats.size()
    agg = torch.zeros(d, device=neighbor_feats.device)
    for q, w in zip(quantiles, weights):
        idx = int(np.ceil(q*n)) - 1
        idx = max(idx, 0)
        agg += w * sorted_feats[idx]
    return agg


class DilatedGCNLayer(nn.Module):
    """
    Provides an option for linear skipping vs. exponential skipping:
      - If exponential_dilation=True, skip_step = m_i * 2^(layer_idx - 1).
      - Else skip_step = m_i.
    threshold T => if neighbors> T, we skip, else no skip (skip_step=1).
    aggregator => aggregator*(1-self_ratio)+ self_ratio*x[i].
    """
    def __init__(self,
                 in_dim, out_dim, layer_idx,
                 T=5, k=2, exponential_dilation=False,
                 quantiles=None, weights=None,
                 use_self_feature=True, self_ratio=0.5,
                 residual=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_idx = layer_idx
        self.T = T
        self.k = k
        self.exponential_dilation = exponential_dilation
        self.quantiles = quantiles if quantiles else [0.25, 0.5, 0.75]
        self.weights = weights
        self.use_self_feature = use_self_feature
        self.self_ratio = self_ratio
        self.residual = residual
        self.linear = nn.Linear(in_dim, out_dim)
        self.enable_residual = (residual and (in_dim == out_dim))

    def forward(self, x, A):
        device = x.device
        N = x.size(0)
        agg_buffer = torch.zeros(N, self.in_dim, device=device)

        for i in range(N):
            neighbors = A[i].nonzero(as_tuple=True)[0].tolist()
            neighbors.sort()
            num_nbr = len(neighbors)
            if num_nbr > self.T:
                m_i = int(np.ceil(num_nbr/self.k))
            else:
                m_i = 1

            skip_step = m_i
            if self.exponential_dilation:
                skip_step *= (2**(self.layer_idx-1))

            keep = []
            remove_idx = set()
            idx = skip_step - 1
            while idx < num_nbr:
                remove_idx.add(idx)
                idx += skip_step

            for idx2, nbr in enumerate(neighbors):
                if idx2 not in remove_idx:
                    keep.append(nbr)

            if len(keep) == 0:
                aggregator = torch.zeros(self.in_dim, device=device)
            else:
                aggregator = quantile_aggregation(x[keep],
                                                  self.quantiles, self.weights)

            if self.use_self_feature:
                aggregator = (1. - self.self_ratio)*aggregator + self.self_ratio*x[i]

            agg_buffer[i] = aggregator

        out = self.linear(agg_buffer)
        if self.enable_residual:
            out = out + x
        return out


class DilatedGCN(nn.Module):
    """
    GCN with dilated skipping + quantile aggregator + optional residual.
    'exponential_dilation' and 'T' are passed to each layer.
    """
    def __init__(self,
                 in_dim, hidden_dim, out_dim, num_layers=4,
                 T=5, k=2, exponential_dilation=False,
                 quantiles=None, weights=None,
                 use_self_feature=True, self_ratio=0.5,
                 residual=True):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.act = nn.ReLU()

        # First
        self.layers.append(
            DilatedGCNLayer(
                in_dim, hidden_dim, layer_idx=1,
                T=T, k=k, exponential_dilation=exponential_dilation,
                quantiles=quantiles, weights=weights,
                use_self_feature=use_self_feature, self_ratio=self_ratio,
                residual=(residual and in_dim==hidden_dim)
            )
        )
        # Hidden
        for l_idx in range(2, num_layers):
            self.layers.append(
                DilatedGCNLayer(
                    hidden_dim, hidden_dim, layer_idx=l_idx,
                    T=T, k=k, exponential_dilation=exponential_dilation,
                    quantiles=quantiles, weights=weights,
                    use_self_feature=use_self_feature, self_ratio=self_ratio,
                    residual=residual
                )
            )
        # Final
        self.layers.append(
            DilatedGCNLayer(
                hidden_dim, out_dim, layer_idx=num_layers,
                T=T, k=k, exponential_dilation=exponential_dilation,
                quantiles=quantiles, weights=weights,
                use_self_feature=use_self_feature, self_ratio=self_ratio,
                residual=False
            )
        )

    def forward(self, x, A):
        for i in range(self.num_layers-1):
            x = self.layers[i](x, A)
            x = self.act(x)
        x = self.layers[-1](x, A)
        return x


###############################################################################
# 4) Training + Utility
###############################################################################
def train_model(model, A, X, y, lr=1e-2, epochs=50):
    optimizer=optim.Adam(model.parameters(), lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
    return model

def measure_accuracy(model, A, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X, A)
        preds = torch.argmax(logits, dim=1)
    return (preds==y).float().mean().item()

def average_embedding_similarity(model, A, X):
    model.eval()
    with torch.no_grad():
        emb = model(X, A)
    emb_norm = nn.functional.normalize(emb, dim=1)
    sim_matrix = emb_norm @ emb_norm.T
    N = sim_matrix.size(0)
    mask = torch.triu(torch.ones(N,N,dtype=bool), diagonal=1)
    return sim_matrix[mask].mean().item()


###############################################################################
# 5) Main: Compare (A) Standard, (B) Dilated Linear, (C) Dilated Exponential
#    in both Sparse + Dense bridging, possibly with multiple T
###############################################################################
def run_experiments(seed=42, bridging_values=(1,25), T_values=(2,5),
                    depths=(2,4,6,8,10,12,14), epochs=50):
    """
    For each bridging in bridging_values (e.g. 1 -> sparse, 25 -> dense),
    for each T in T_values,
    for each depth in depths,
    compare:
      - Standard GCN
      - Dilated GCN (linear skipping)
      - Dilated GCN (exponential skipping)

    Prints accuracy + similarity for each config.
    """
    for bridging in bridging_values:
        print(f"\n===== Ring-of-Clusters: bridging={bridging} ===== (seed={seed})")
        A, X, y = generate_ring_of_clusters(
            num_clusters=5,
            cluster_size=20,
            bridging=bridging,
            feature_dim=8,
            seed=seed
        )
        for T in T_values:
            print(f"\n--- Threshold T={T} ---")
            for depth in depths:
                # 1) Standard
                std_model = StandardGCN(in_dim=8, hidden_dim=8, out_dim=5, num_layers=depth)
                train_model(std_model, A, X, y, lr=1e-2, epochs=epochs)
                std_acc = measure_accuracy(std_model, A, X, y)
                std_sim = average_embedding_similarity(std_model, A, X)

                # 2) Dilated, Linear
                dil_linear = DilatedGCN(
                    in_dim=8, hidden_dim=8, out_dim=5, num_layers=depth,
                    T=T, k=2, exponential_dilation=False,
                    quantiles=[0.25,0.5,0.75], weights=None,
                    use_self_feature=True, self_ratio=0.5,
                    residual=True
                )
                train_model(dil_linear, A, X, y, lr=1e-2, epochs=epochs)
                dl_acc = measure_accuracy(dil_linear, A, X, y)
                dl_sim = average_embedding_similarity(dil_linear, A, X)

                # 3) Dilated, Exponential
                dil_exp = DilatedGCN(
                    in_dim=8, hidden_dim=8, out_dim=5, num_layers=depth,
                    T=T, k=2, exponential_dilation=True,
                    quantiles=[0.25,0.5,0.75], weights=None,
                    use_self_feature=True, self_ratio=0.5,
                    residual=True
                )
                train_model(dil_exp, A, X, y, lr=1e-2, epochs=epochs)
                de_acc = measure_accuracy(dil_exp, A, X, y)
                de_sim = average_embedding_similarity(dil_exp, A, X)

                print(f"Depth={depth:2d} |"
                      f" StdAcc={std_acc:.3f} StdSim={std_sim:.3f} |"
                      f" Dil(Lin)Acc={dl_acc:.3f} LinSim={dl_sim:.3f} |"
                      f" Dil(Exp)Acc={de_acc:.3f} ExpSim={de_sim:.3f}")


def main():
    print("Comparing Standard vs. Dilated (Linear & Exponential) on ring-of-clusters.\n"
          "We also vary T in [2,5]. bridging in [1,25]. Depth in [2,4,6,8,10,12,14].\n")
    run_experiments(seed=42,
                    bridging_values=(1,25),
                    T_values=(2,5),
                    depths=(2,4,6,8,10,12,14),
                    epochs=50)

if __name__=="__main__":
    main()

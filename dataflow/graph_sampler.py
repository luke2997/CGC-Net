import numpy as np
import torch
from sklearn.neighbors import KDTree

def frequency_adaptive_graph2(
    coords, freq, density, freq_var=None, 
    choice=None, alpha=0.5, num_freq_bins=4, num_density_bins=4,
    eps0=2.0, delta_f0=0.5, gamma_rho=0.5, gamma_sigma=0.5,
    n_sample=4, loop=False, sparse=False, random_seed=42
):
    np.random.seed(random_seed)

    if choice is not None:
        coords = coords[choice]
        freq = freq[choice]
        density = density[choice]
        if freq_var is not None:
            freq_var = freq_var[choice]

    N = len(coords)
    if freq_var is None:
        freq_var = np.zeros(N)

    # -----------------------------
    # 1) Frequency-aware sampling
    # -----------------------------
    freq_edges = np.percentile(freq, np.linspace(0, 100, num_freq_bins+1))
    density_edges = np.percentile(density, np.linspace(0, 100, num_density_bins+1))

    freq_bin_id = np.digitize(freq, freq_edges) - 1
    freq_bin_id = np.clip(freq_bin_id, 0, num_freq_bins - 1)
    density_bin_id = np.digitize(density, density_edges) - 1
    density_bin_id = np.clip(density_bin_id, 0, num_density_bins - 1)

    bin_dict = {}
    for i in range(N):
        bin_dict.setdefault((density_bin_id[i], freq_bin_id[i]), []).append(i)

    sampled_indices = []
    for bin_pair, node_list in bin_dict.items():
        bin_size = len(node_list)
        target_count = int(np.round(alpha * bin_size))
        if target_count > 0:
            chosen = np.random.choice(node_list, size=target_count, replace=False)
            sampled_indices.extend(chosen)
    sampled_indices = np.array(sampled_indices, dtype=int)

    # reorder everything
    coords_sub = coords[sampled_indices]
    freq_sub = freq[sampled_indices]
    density_sub = density[sampled_indices]
    var_sub = freq_var[sampled_indices]
    N_sub = len(coords_sub)

    # -----------------------------
    # 2) Adaptive threshold adjacency 
    # -----------------------------
    eps_array = eps0 / (1 + gamma_rho * density_sub)
    df_array  = delta_f0 / (1 + gamma_sigma * var_sub)

    kdtree = KDTree(coords_sub)
    eps_max = eps_array.max()

    adj_matrix = np.zeros((N_sub, N_sub), dtype=float)

    for i in range(N_sub):
        xi = coords_sub[i]
        fi = freq_sub[i]
        eps_i = eps_array[i]
        df_i  = df_array[i]
        # radius search up to eps_max
        idx_spatial = kdtree.query_radius(xi.reshape(1, -1), r=eps_max)[0]
        for j in idx_spatial:
            if j == i:
                continue
            dist_xy = np.linalg.norm(coords_sub[j] - xi)
            if dist_xy <= eps_i:
                if abs(freq_sub[j] - fi) <= df_i:
                    adj_matrix[i, j] = 1.0

    # -----------------------------
    # 3) Random edge selection 
    # -----------------------------
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1e-12
    prob_matrix = adj_matrix / row_sums
    cdf_matrix = np.cumsum(prob_matrix, axis=1)

    final_adj = np.zeros((N_sub, N_sub), dtype=float)
    # vectorized approach: draw (N_sub * n_sample) random numbers
    rand_vals = np.random.rand(N_sub, n_sample)
    for i in range(N_sub):
        chosen_cols = np.searchsorted(cdf_matrix[i], rand_vals[i], side='right')
        final_adj[i, chosen_cols] = 1
        final_adj[chosen_cols, i] = 1

    if not loop:
        np.fill_diagonal(final_adj, 0.0)

    if sparse:
        row_idx, col_idx = np.nonzero(final_adj)
        row_t = torch.from_numpy(row_idx).long()
        col_t = torch.from_numpy(col_idx).long()
        return torch.stack([row_t, col_t], dim=0)
    else:
        return torch.from_numpy(final_adj)

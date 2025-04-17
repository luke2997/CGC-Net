# fmt: off
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.dense.linear import Linear as DenseLinear
from torch_geometric.utils import to_dense_batch

try:
    from model.utils import to_dense_adj
except ImportError:
    print("Warning: model.utils.to_dense_adj not found. Falling back to torch_geometric.utils.to_dense_adj.")
    from torch_geometric.utils import to_dense_adj
from torch.nn import LSTM
import math
from typing import List, Optional, Tuple, Union, Any

EPS = 1e-15
class QuantileDilatedDenseSAGEConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 taus: List[float] = [0.25, 0.5, 0.75],
                 weights: List[float] = [0.25, 0.5, 0.25],
                 T: int = 10,
                 k: int = 2,
                 layer_idx: int = 1,
                 dilation_mode: str = 'linear',
                 normalize: bool = True, # Default True per user request
                 bias: bool = True,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.T = T
        self.k = k
        self.layer_idx = layer_idx
        assert dilation_mode in ['linear', 'exponential'], \
            "dilation_mode must be either 'linear' or 'exponential'"
        self.dilation_mode = dilation_mode
        self.normalize = normalize
        self.bias = bias

        # --- Quantile Parameters ---
        assert len(taus) == len(weights), "taus and weights must have the same length"
        assert abs(sum(weights) - 1.0) < EPS, "weights must sum to 1"
        # Register taus first
        self.register_buffer('taus', torch.tensor(taus, dtype=torch.float))
        # Register weights ensuring dtype matches taus (Correct based on previous revision)
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float).to(self.taus.dtype))
        self.num_quantiles = len(taus)

        # --- SAGE Linear Layers ---
        self.lin_l = DenseLinear(in_channels, out_channels, bias=bias)
        self.lin_r = DenseLinear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, adj, mask=None, add_loop=True):
        """
        Args:
            x (Tensor): Node feature tensor :math:`(|\mathcal{B}| \times N \times F_{in})`.
            adj (Tensor): Adjacency tensor :math:`(|\mathcal{B}| \times N \times N)`.
            mask (BoolTensor or FloatTensor, optional): Mask matrix
                :math:`(|\mathcal{B}| \times N)`, indicating valid nodes. (default: :obj:`None`)
            add_loop (bool): If True, add self-loops before dilation/aggregation.

        Returns:
            Tensor: Output node features :math:`(|\mathcal{B}| \times N \times F_{out}`.
        """
        batch_size, num_nodes, _ = x.shape
        dev, dtp = x.device, x.dtype

        if add_loop:
            idx = torch.arange(num_nodes, device=dev)
            adj_ = adj.clone()
            adj_[:, idx, idx] = 1.0
        else:
            adj_ = adj

        # --- Dilation Step ---
        adj_dilated = adj_.clone()
        # Loop remains necessary for dilation logic as written
        for b in range(batch_size):
            for n in range(num_nodes):
                # Mask usage correct for [B, N] shape
                if mask is not None and not mask[b, n]:
                    continue
                neighbors_indices = (adj_[b, n] > 0).nonzero(as_tuple=False).view(-1)
                is_self = neighbors_indices == n
                if is_self.any():
                    neighbors_wo_self = neighbors_indices[~is_self]
                    num_neighbors = len(neighbors_wo_self)
                else:
                    neighbors_wo_self = neighbors_indices
                    num_neighbors = len(neighbors_wo_self)
                if num_neighbors == 0: continue
                if num_neighbors > self.T: skip_factor = math.ceil(num_neighbors / self.k)
                else: skip_factor = 1
                if self.dilation_mode == 'exponential': skip_factor *= (2**(self.layer_idx - 1))
                skip_factor = max(1, int(skip_factor))
                if skip_factor > 1:
                    to_remove_relative_idx = torch.arange(skip_factor - 1, num_neighbors, skip_factor, device=dev)
                    if to_remove_relative_idx.numel() > 0:
                        to_remove_global_idx = neighbors_wo_self[to_remove_relative_idx]
                        adj_dilated[b, n, to_remove_global_idx] = 0.0
                        adj_dilated[b, to_remove_global_idx, n] = 0.0

        N = num_nodes
        # Expand features and create NaN mask based on dilated adjacency
        x_exp = x.unsqueeze(1).expand(-1, N, -1, -1) # [B, N_target, N_source, C_in]
        # Use adj_dilated for neighbor definition
        x_mask = torch.where(
            adj_dilated.unsqueeze(-1) > 0, # Check for neighbors based on dilated adj
            x_exp,
            torch.tensor(float('nan'), device=dev, dtype=dtp) # Use input dtype/device for nan
        )

        # Compute quantiles along the source neighbor dimension (dim=2)
        q = torch.nanquantile(
            x_mask,
            q=self.taus.to(dtype=dtp, device=dev), # Ensure taus match input dtype/device
            dim=2, # Aggregate over source dimension N
            interpolation='linear'
        )
        # Compute weighted sum of quantiles
        weights_b = self.weights.to(dtype=dtp, device=dev).view(1, 1, -1, 1) # [1, 1, M, 1]
        aggregated_features = torch.sum(q * weights_b, dim=2) # Sum over M dimension (dim 2)
        # aggregated_features shape: [B, N_target, C_in]

        # Handle nodes with no neighbors (replace NaN with 0)
        aggregated_features = torch.nan_to_num(aggregated_features, nan=0.0)

        out_neighbors = self.lin_l(aggregated_features)
        out_self = self.lin_r(x)
        out = out_neighbors + out_self
        
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        if mask is not None:
             out = out * mask.unsqueeze(-1).to(dtp)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, taus={self.taus.tolist()}, '
                f'normalize={self.normalize})')


class QuantileDilatedDenseGINConv(nn.Module):
    def __init__(self,
                 nn: Any, # Should be torch.nn.Module
                 eps: float = 0.,
                 train_eps: bool = False,
                 taus: List[float] = [0.25, 0.5, 0.75],
                 weights: List[float] = [0.25, 0.5, 0.25],
                 T: int = 10,
                 k: int = 2,
                 layer_idx: int = 1,
                 dilation_mode: str = 'linear',
                 **kwargs):
        super().__init__()

        self.nn = nn
        self.initial_eps = eps
        if train_eps: self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else: self.register_buffer('eps', torch.Tensor([eps]))

        self.T = T; self.k = k; self.layer_idx = layer_idx
        assert dilation_mode in ['linear', 'exponential']
        self.dilation_mode = dilation_mode

        # Quantile Parameters (Consistent with SAGE)
        assert len(taus) == len(weights) and abs(sum(weights) - 1.0) < EPS
        self.register_buffer('taus', torch.tensor(taus, dtype=torch.float))
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float).to(self.taus.dtype))
        self.num_quantiles = len(taus)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.nn, 'reset_parameters'): self.nn.reset_parameters()
        with torch.no_grad():
            if isinstance(self.eps, nn.Parameter): self.eps.fill_(self.initial_eps)
            # else buffer is already set

    def forward(self, x, adj, mask=None, add_loop=True):
        batch_size, num_nodes, C_in = x.shape
        dev, dtp = x.device, x.dtype

        if add_loop:
            idx = torch.arange(num_nodes, device=dev)
            adj_ = adj.clone(); adj_[:, idx, idx] = 1.0
        else: adj_ = adj

        # Dilation Step (Identical to SAGE)
        adj_dilated = adj_.clone()
        for b in range(batch_size):
            for n in range(num_nodes):
                if mask is not None and not mask[b, n]: continue
                neighbors_indices = (adj_[b, n] > 0).nonzero(as_tuple=False).view(-1)
                is_self = neighbors_indices == n
                if is_self.any(): neighbors_wo_self = neighbors_indices[~is_self]
                else: neighbors_wo_self = neighbors_indices
                num_neighbors = len(neighbors_wo_self)
                if num_neighbors == 0: continue
                if num_neighbors > self.T: skip_factor = math.ceil(num_neighbors / self.k)
                else: skip_factor = 1
                if self.dilation_mode == 'exponential': skip_factor *= (2**(self.layer_idx - 1))
                skip_factor = max(1, int(skip_factor))
                if skip_factor > 1:
                    to_remove_relative_idx = torch.arange(skip_factor - 1, num_neighbors, skip_factor, device=dev)
                    if to_remove_relative_idx.numel() > 0:
                       to_remove_global_idx = neighbors_wo_self[to_remove_relative_idx]
                       adj_dilated[b, n, to_remove_global_idx] = 0.0
                       adj_dilated[b, to_remove_global_idx, n] = 0.0

        # Vectorized Quantile Aggregation Step (Identical to SAGE)
        N = num_nodes
        x_exp = x.unsqueeze(1).expand(-1, N, -1, -1)
        x_mask = torch.where(adj_dilated.unsqueeze(-1) > 0, x_exp,
                             torch.tensor(float('nan'), device=dev, dtype=dtp))
        q = torch.nanquantile(x_mask, q=self.taus.to(dtype=dtp, device=dev), dim=2, interpolation='linear')
        weights_b = self.weights.to(dtype=dtp, device=dev).view(1, 1, -1, 1)
        aggregated_features = torch.sum(q * weights_b, dim=2)
        aggregated_features = torch.nan_to_num(aggregated_features, nan=0.0)

        # GIN Combination Step
        out = (1 + self.eps.to(dtp)) * x + aggregated_features

        # Apply MLP
        out = self.nn(out)

        # Apply mask (Requires unsqueeze for [B, N] mask)
        if mask is not None:
             out = out * mask.unsqueeze(-1).to(dtp)

        return out

    def __repr__(self) -> str:
        eps_val = self.eps.item() if isinstance(self.eps, torch.Tensor) else self.eps
        return (f'{self.__class__.__name__}(nn={self.nn}, '
                f'eps={eps_val:.4f}, taus={self.taus.tolist()})')


########################################################
# Jumping Knowledge Aggregation (Unchanged)
########################################################
class DenseJK(nn.Module):
    r""" Aggregates representations across different layers (LSTM-based JK). """
    def __init__(self, mode, channels=None, num_layers=None):
        super(DenseJK, self).__init__()
        self.channel = channels
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']
        if mode == 'lstm':
            assert channels is not None and num_layers is not None
            lstm_hidden_dim = channels
            self.lstm = LSTM(input_size=channels, hidden_size=lstm_hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.att = DenseLinear(2 * lstm_hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'): self.lstm.reset_parameters()
        if hasattr(self, 'att'): self.att.reset_parameters()

    def forward(self, xs: Union[List[torch.Tensor], Tuple[torch.Tensor, ...], torch.Tensor]):
        if self.mode == 'cat':
            if isinstance(xs, (list, tuple)): return torch.cat(xs, dim=-1)
            else: return xs
        if self.mode == 'max':
            if isinstance(xs, (list, tuple)):
                 return torch.max(torch.stack(xs, dim=0), dim=0)[0]
            else:
                 assert self.channel is not None and xs.shape[-1] % self.channel == 0
                 xs_split = torch.split(xs, self.channel, dim=-1)
                 return torch.max(torch.stack(xs_split, dim=0), dim=0)[0]
        if self.mode == 'lstm':
            if not isinstance(xs, (list, tuple)):
                assert self.channel is not None and xs.shape[-1] % self.channel == 0
                xs_list = torch.split(xs, self.channel, dim=-1)
                xs_stack = torch.stack(xs_list, dim=2)
            else: xs_stack = torch.stack(xs, dim=2)
            shape = xs_stack.shape
            x_lstm_in = xs_stack.reshape(-1, shape[2], shape[3])
            alpha, _ = self.lstm(x_lstm_in)
            att_scores = self.att(alpha).squeeze(-1)
            att_weights = torch.softmax(att_scores, dim=-1)
            out = (x_lstm_in * att_weights.unsqueeze(-1)).sum(dim=1)
            out = out.reshape(shape[0], shape[1], shape[3])
            return out
        return xs # Should not be reached

    def __repr__(self):
        return f'{self.__class__.__name__}(mode={self.mode}, channels={self.channel})'

########################################################
# GNN Module (Updated for Quantile Layers and keyword args)
########################################################
class GNN_Module(nn.Module):
    r"""
    A 3-layer block using QuantileDilatedDenseSAGEConv or QuantileDilatedDenseGINConv.
    Followed by BN + activation. Outputs concatenated features (unless lin=True).
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim,
                 bias, bn, add_loop,
                 lin=False, # Defaulting lin=False for JK
                 gcn_name='SAGE', sync=False, activation='relu',
                 jk=False, # Retained flag
                 # Dilation parameters
                 T=10, k=2, layer_idx_start=1, dilation_mode='linear',
                 # Quantile parameters
                 taus=[0.25, 0.5, 0.75], weights=[0.25, 0.5, 0.25],
                 # Layer specific flags
                 normalize_sage=True, train_eps_gin=False):
        super(GNN_Module, self).__init__()
        self.add_loop = add_loop
        self.activation_name = activation
        self.gcn_name = gcn_name
        self.taus = taus; self.weights = weights
        self.normalize_sage = normalize_sage
        self.train_eps_gin = train_eps_gin

        # Instantiate layers using the _gcn helper
        self.gcn1 = self._gcn(gcn_name, input_dim, hidden_dim, bias, T, k, layer_idx_start, dilation_mode)
        self.active1 = self._activation(activation)
        self.gcn2 = self._gcn(gcn_name, hidden_dim, hidden_dim, bias, T, k, layer_idx_start+1, dilation_mode)
        self.active2 = self._activation(activation)
        self.gcn3 = self._gcn(gcn_name, hidden_dim, embedding_dim, bias, T, k, layer_idx_start+2, dilation_mode)
        self.active3 = self._activation(activation)

        self.bn_flag = bn
        if bn:
            bn_class = nn.SyncBatchNorm if sync else nn.BatchNorm1d
            self.bn1 = bn_class(hidden_dim)
            self.bn2 = bn_class(hidden_dim)
            self.bn3 = bn_class(embedding_dim)

        self.lin_flag = lin
        if lin:
            concat_dim = hidden_dim + hidden_dim + embedding_dim
            self.lin = DenseLinear(concat_dim, embedding_dim)
        else: self.lin = None

    def _activation(self, name='relu'):
        inplace = True
        if name == 'relu': return nn.ReLU(inplace=inplace)
        elif name == 'elu': return nn.ELU(inplace=inplace)
        elif name == 'leakyrelu': return nn.LeakyReLU(inplace=inplace)
        else: raise ValueError(f"Unsupported activation: {name}")

    def _gcn(self, name, input_dim, output_dim, bias, T, k, layer_idx, dilation_mode):
        if name.upper() == 'SAGE':
            return QuantileDilatedDenseSAGEConv(
                in_channels=input_dim, out_channels=output_dim, bias=bias,
                taus=self.taus, weights=self.weights, T=T, k=k, layer_idx=layer_idx,
                dilation_mode=dilation_mode, normalize=self.normalize_sage)
        elif name.upper() == 'GIN':
            nn_mlp = nn.Sequential(
                DenseLinear(input_dim, output_dim), self._activation(self.activation_name),
                DenseLinear(output_dim, output_dim))
            return QuantileDilatedDenseGINConv(
                nn=nn_mlp, eps=0., train_eps=self.train_eps_gin,
                taus=self.taus, weights=self.weights, T=T, k=k, layer_idx=layer_idx,
                dilation_mode=dilation_mode)
        else: raise ValueError(f"Unsupported GCN type: {name}")

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels); x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels); return x

    def forward(self, x, adj, mask=None):
        # Keyword args used (Correct based on previous revision)
        x1 = self.gcn1(x, adj, mask=mask, add_loop=self.add_loop)
        x1 = self.active1(x1); x1 = self.bn(1, x1) if self.bn_flag else x1
        x2 = self.gcn2(x1, adj, mask=mask, add_loop=self.add_loop)
        x2 = self.active2(x2); x2 = self.bn(2, x2) if self.bn_flag else x2
        x3 = self.gcn3(x2, adj, mask=mask, add_loop=self.add_loop)
        x3 = self.active3(x3); x3 = self.bn(3, x3) if self.bn_flag else x3

        x_cat = torch.cat([x1, x2, x3], dim=-1)

        # Mask usage correct for [B, N] shape requires unsqueeze
        if mask is not None: x_cat = x_cat * mask.unsqueeze(-1).to(x_cat.dtype)
        if self.lin_flag:
            x_cat = self.lin(x_cat)
            if mask is not None: x_cat = x_cat * mask.unsqueeze(-1).to(x_cat.dtype)
        return x_cat


########################################################
# Full Soft Pooling GCN (Updates integrated)
########################################################
class SoftPoolingGcnEncoder(nn.Module):
    r""" Hierarchical (DiffPool-like) GNN using Quantile Dilated Convolutions. """
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim,
                 bias, bn, assign_hidden_dim, label_dim,
                 assign_ratio=0.25, pred_hidden_dims=[50],
                 gcn_name='SAGE', collect_assign=False, load_data_sparse=False,
                 norm_adj=False, activation='relu', drop_out=0.,
                 jk_mode=None, # Default None
                 # Dilation parameters
                 T=10, k=2, dilation_mode='linear',
                 # Quantile parameters
                 taus=[0.25, 0.5, 0.75], weights=[0.25, 0.5, 0.25],
                 # Layer specific flags
                 normalize_sage=True, train_eps_gin=False):
        super(SoftPoolingGcnEncoder, self).__init__()
        self.jk_mode = jk_mode.lower() if isinstance(jk_mode, str) else None
        self.drop_out = drop_out; self.norm_adj = norm_adj
        self.load_data_sparse = load_data_sparse; self.collect_assign = collect_assign
        self.assign_matrix = []; self.gcn_name = gcn_name

        num_assign_nodes_1 = int(max_num_nodes * assign_ratio)
        num_assign_nodes_2 = int(num_assign_nodes_1 * assign_ratio)

        # Common args passed via dictionaries
        embed_gnn_args = dict(bias=bias, bn=bn, add_loop=False, lin=False,
                              gcn_name=gcn_name, activation=activation, T=T, k=k,
                              dilation_mode=dilation_mode, taus=taus, weights=weights,
                              normalize_sage=normalize_sage, train_eps_gin=train_eps_gin)
        pool_gnn_args = dict(bias=bias, bn=bn, add_loop=False, lin=True,
                             gcn_name=gcn_name, activation=activation, T=T, k=k,
                             dilation_mode=dilation_mode, taus=taus, weights=weights,
                             normalize_sage=normalize_sage, train_eps_gin=train_eps_gin)

        # Stage 1 (Duplicate GCN_embed_1 removed based on previous revision)
        self.GCN_embed_1 = GNN_Module(input_dim, hidden_dim, embedding_dim, layer_idx_start=1, **embed_gnn_args)
        gnn1_out_dim = 2 * hidden_dim + embedding_dim
        self.jk1 = DenseJK(self.jk_mode, channels=gnn1_out_dim, num_layers=1) if self.jk_mode else None
        self.GCN_pool_1 = GNN_Module(input_dim, assign_hidden_dim, num_assign_nodes_1, layer_idx_start=1, **pool_gnn_args)

        input_dim_second = gnn1_out_dim # Adjust if JK changes dim

        # Stage 2
        self.GCN_embed_2 = GNN_Module(input_dim_second, hidden_dim, embedding_dim, layer_idx_start=4, **embed_gnn_args)
        gnn2_out_dim = 2 * hidden_dim + embedding_dim
        self.jk2 = DenseJK(self.jk_mode, channels=gnn2_out_dim, num_layers=1) if self.jk_mode else None
        self.GCN_pool_2 = GNN_Module(input_dim_second, assign_hidden_dim, num_assign_nodes_2, layer_idx_start=4, **pool_gnn_args)

        input_dim_third = gnn2_out_dim # Adjust if JK changes dim

        # Stage 3
        self.GCN_embed_3 = GNN_Module(input_dim_third, hidden_dim, embedding_dim, layer_idx_start=7, **embed_gnn_args)
        gnn3_out_dim = 2 * hidden_dim + embedding_dim
        self.jk3 = DenseJK(self.jk_mode, channels=gnn3_out_dim, num_layers=1) if self.jk_mode else None

        # Final Readout MLP
        pred_input_dim = gnn1_out_dim + gnn2_out_dim + gnn3_out_dim # Adjust if JK changes dim
        self.pred_model = self.build_readout_module(pred_input_dim, pred_hidden_dims, label_dim, activation)
        self.init_weights()

    def init_weights(self):
        for module in self.pred_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None: nn.init.zeros_(module.bias)

    @staticmethod
    def construct_mask(max_nodes, batch_num_nodes):
        # Returns [B, N] float mask (Correct based on previous revision)
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes, dtype=torch.float, device=batch_num_nodes.device)
        for i, num in enumerate(batch_num_nodes):
             idx = int(num)
             if idx > 0: out_tensor[i, :idx] = 1.0
        return out_tensor

    def _re_norm_adj(self, adj, p=0.4, mask=None):
        # Uses [B, N] mask correctly (Correct based on previous revision)
        batch_size, num_nodes, _ = adj.shape
        new_adj = adj.clone(); idx = torch.arange(num_nodes, device=adj.device)
        new_adj[:, idx, idx] = 0
        row_sum = new_adj.sum(dim=-1, keepdim=True)
        normalized_off_diag = torch.div(new_adj, row_sum + EPS) * (1 - p)
        normalized_off_diag[row_sum == 0] = 0
        normalized_off_diag[:, idx, idx] = p
        if mask is not None:
            mask_unsqueezed = mask.unsqueeze(-1)
            mask_2d = mask_unsqueezed @ mask_unsqueezed.transpose(1, 2)
            normalized_off_diag = normalized_off_diag * mask_2d.to(normalized_off_diag.dtype)
        return normalized_off_diag

    def _diff_pool(self, x, adj, s, mask=None):
        # Handles [B, N] mask correctly (Correct based on previous revision)
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s, dim=-1)
        if self.collect_assign: self.assign_matrix.append(s.detach().cpu())
        if mask is not None: s = s * mask.unsqueeze(-1).to(s.dtype) # Unsqueeze mask
        x_pool = torch.matmul(s.transpose(1, 2), x)
        adj_pool = torch.matmul(s.transpose(1, 2), torch.matmul(adj, s))
        return x_pool, adj_pool

    def _activation(self, name='relu'):
        inplace = True
        if name == 'relu': return nn.ReLU(inplace=inplace)
        elif name == 'elu': return nn.ELU(inplace=inplace)
        elif name == 'leakyrelu': return nn.LeakyReLU(inplace=inplace)
        else: raise ValueError(f"Unsupported activation: {name}")

    def build_readout_module(self, pred_input_dim, pred_hidden_dims, label_dim, activation):
        pred_layers = []; current_dim = pred_input_dim
        for hidden_dim in pred_hidden_dims:
            pred_layers.append(DenseLinear(current_dim, hidden_dim))
            pred_layers.append(self._activation(activation))
            if self.drop_out > 0: pred_layers.append(nn.Dropout(self.drop_out))
            current_dim = hidden_dim
        pred_layers.append(DenseLinear(current_dim, label_dim))
        return nn.Sequential(*pred_layers)

    def _sparse_to_dense_input(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_num_nodes = torch.bincount(batch); max_num_nodes = batch_num_nodes.max().item()
        device = x.device
        dense_x, _ = to_dense_batch(x, batch, max_num_nodes=max_num_nodes) # Ignore bool mask from here
        dense_adj = to_dense_adj(edge_index, batch, max_num_nodes=max_num_nodes)
        batch_num_nodes = batch_num_nodes.to(device)
        label = data.y.to(device) if hasattr(data, 'y') and data.y is not None else None
        return dense_x, dense_adj, batch_num_nodes, label # Return counts

    def forward(self, data):
        self.assign_matrix = []
        if self.load_data_sparse:
            x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes = data[0], data[1], data[2]
            label = data[3] if len(data) > 3 else None
            if not isinstance(batch_num_nodes, torch.Tensor): batch_num_nodes = torch.tensor(batch_num_nodes, device=x.device)
            else: batch_num_nodes = batch_num_nodes.to(x.device) # Ensure device

        max_num_nodes = adj.size(1)
        # Create [B, N] float mask (Correct based on previous revision)
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes).to(x.device)

        if self.norm_adj: adj = self._re_norm_adj(adj, p=0.4, mask=embedding_mask)

        out_all = []

        # Stage 1
        embed_1 = self.GCN_embed_1(x, adj, embedding_mask) # Pass [B, N] mask
        if self.jk1: embed_1 = self.jk1(embed_1)
        # Max pool requires unsqueezed mask
        out_1, _ = torch.max(embed_1 * embedding_mask.unsqueeze(-1).to(embed_1.dtype), dim=1)
        out_all.append(out_1)
        assign_1 = self.GCN_pool_1(x, adj, embedding_mask) # Pass [B, N] mask
        x_pool_1, adj_pool_1 = self._diff_pool(embed_1, adj, assign_1, embedding_mask)

        # Stage 2
        if self.norm_adj: adj_pool_1 = self._re_norm_adj(adj_pool_1, p=0.4, mask=None)
        embed_2 = self.GCN_embed_2(x_pool_1, adj_pool_1, mask=None)
        if self.jk2: embed_2 = self.jk2(embed_2)
        out_2, _ = torch.max(embed_2, dim=1); out_all.append(out_2)
        assign_2 = self.GCN_pool_2(x_pool_1, adj_pool_1, mask=None)
        x_pool_2, adj_pool_2 = self._diff_pool(embed_2, adj_pool_1, assign_2, mask=None)

        # Stage 3
        if self.norm_adj: adj_pool_2 = self._re_norm_adj(adj_pool_2, p=0.4, mask=None)
        embed_3 = self.GCN_embed_3(x_pool_2, adj_pool_2, mask=None)
        if self.jk3: embed_3 = self.jk3(embed_3)
        out_3, _ = torch.max(embed_3, dim=1); out_all.append(out_3)

        # Final Prediction
        final_graph_embedding = torch.cat(out_all, dim=1)
        if self.drop_out > 0: final_graph_embedding = F.dropout(final_graph_embedding, p=self.drop_out, training=self.training)
        output_logits = self.pred_model(final_graph_embedding)

        if self.training and label is not None:
            cls_loss = F.cross_entropy(output_logits, label)
            return output_logits, cls_loss
        elif not self.training and label is not None:
             cls_loss = F.cross_entropy(output_logits, label)
             return output_logits, cls_loss
        else: return output_logits
# fmt: on

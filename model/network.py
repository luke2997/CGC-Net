import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseSAGEConv, DenseGINConv
from torch_geometric.utils import to_dense_batch
from model.utils import to_dense_adj
from torch.nn import Linear, LSTM
import math

EPS = 1e-15

########################################################
# Dilated Convolution Classes
########################################################

class DilatedDenseSAGEConv(DenseSAGEConv):
    r"""
    Extends DenseSAGEConv with dilated neighbor skipping (linear or exponential).
    
    Attributes:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        T (int): Threshold parameter for neighborhood size.
        k (int): Hyperparameter controlling how many neighbors to skip.
        layer_idx (int): Current layer index (1-based).
        dilation_mode (str): 'linear' or 'exponential'.
        
    Usage:
        - If dilation_mode='linear', skip factor = m_i (above).
        - If dilation_mode='exponential', skip factor = m_i * 2^(layer_idx-1).
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 T=10,
                 k=2,
                 layer_idx=1,
                 dilation_mode='linear',
                 **kwargs):
        super(DilatedDenseSAGEConv, self).__init__(in_channels, out_channels, **kwargs)
        self.T = T
        self.k = k
        self.layer_idx = layer_idx
        assert dilation_mode in ['linear', 'exponential'], \
            "dilation_mode must be either 'linear' or 'exponential'"
        self.dilation_mode = dilation_mode

    def forward(self, x, adj, mask=None, add_loop=False):
        """
        1) Determine skip factor m_i for each node i in the batch.
        2) Remove every m_i-th neighbor from adjacency before calling the parent DenseSAGEConv.
        """
        # x: [batch_size, num_nodes, in_channels]
        # adj: [batch_size, num_nodes, num_nodes]
        # mask: [batch_size, num_nodes] (optional)
        #----------------------------------------
        batch_size, num_nodes, _ = x.shape
        if add_loop:
            idx = torch.arange(num_nodes, device=adj.device)
            adj[:, idx, idx] = 1.0

        # Build a new adjacency that has some neighbors removed:
        adj_dilated = adj.clone()

        for b in range(batch_size):
            for n in range(num_nodes):
                if mask is not None and mask[b, n] == 0:
                    # If node is masked out, skip
                    continue

                # Identify neighbors of node n
                # "neighbors" is a 1D boolean or float array of shape (num_nodes,)
                neighbors = (adj_dilated[b, n] > 0).nonzero(as_tuple=False).view(-1)
                num_neighbors = len(neighbors)
                if num_neighbors <= 1:
                    continue

                # skip factor m_i
                if num_neighbors > self.T:
                    # linear portion
                    skip_factor = math.ceil(num_neighbors / self.k)
                else:
                    skip_factor = 1

                if self.dilation_mode == 'exponential':
                    # multiply by 2^(layer_idx - 1)
                    skip_factor *= (2 ** (self.layer_idx - 1))

                # Now remove every skip_factor-th neighbor
                # (Indices in "neighbors" are sorted in ascending order)
                to_remove = neighbors[ skip_factor-1 :: skip_factor ]
                # "skip_factor-1" is to say "start from the skip_factor-th neighbor"
                # then take every skip_factor-th after that.

                # Zero out edges in adjacency for those neighbors
                adj_dilated[b, n, to_remove] = 0.0
                adj_dilated[b, to_remove, n] = 0.0  # keep adjacency symmetric

        # Now call standard DenseSAGEConv with the modified adjacency
        return super(DilatedDenseSAGEConv, self).forward(x, adj_dilated, mask, add_loop)


class DilatedDenseGINConv(DenseGINConv):
    r"""
    Extends DenseGINConv with dilated neighbor skipping (linear or exponential).
    """
    def __init__(self,
                 nn,
                 T=10,
                 k=2,
                 layer_idx=1,
                 dilation_mode='linear',
                 **kwargs):
        super(DilatedDenseGINConv, self).__init__(nn, **kwargs)
        self.T = T
        self.k = k
        self.layer_idx = layer_idx
        assert dilation_mode in ['linear', 'exponential'], \
            "dilation_mode must be either 'linear' or 'exponential'"
        self.dilation_mode = dilation_mode

    def forward(self, x, adj, mask=None, add_loop=False):
        batch_size, num_nodes, _ = x.shape
        if add_loop:
            idx = torch.arange(num_nodes, device=adj.device)
            adj[:, idx, idx] = 1.0

        adj_dilated = adj.clone()
        for b in range(batch_size):
            for n in range(num_nodes):
                if mask is not None and mask[b, n] == 0:
                    continue

                neighbors = (adj_dilated[b, n] > 0).nonzero(as_tuple=False).view(-1)
                num_neighbors = len(neighbors)
                if num_neighbors <= 1:
                    continue

                if num_neighbors > self.T:
                    skip_factor = math.ceil(num_neighbors / self.k)
                else:
                    skip_factor = 1

                if self.dilation_mode == 'exponential':
                    skip_factor *= (2 ** (self.layer_idx - 1))

                to_remove = neighbors[ skip_factor-1 :: skip_factor ]
                adj_dilated[b, n, to_remove] = 0.0
                adj_dilated[b, to_remove, n] = 0.0

        return super(DilatedDenseGINConv, self).forward(x, adj_dilated, mask, add_loop)

########################################################
# Jumping Knowledge Aggregation
########################################################

class DenseJK(nn.Module):
    r"""
    Aggregates representations across different layers (LSTM-based JK).
    """
    def __init__(self, mode, channels=None, num_layers=None):
        super(DenseJK, self).__init__()
        self.channel = channels
        self.mode = mode.lower()
        assert self.mode in ['cat', 'max', 'lstm']

        if mode == 'lstm':
            assert channels is not None
            assert num_layers is not None
            self.lstm = LSTM(
                channels,
                channels * num_layers // 2,
                bidirectional=True,
                batch_first=True)
            self.att = Linear(2 * channels * num_layers // 2, 1)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lstm'):
            self.lstm.reset_parameters()
        if hasattr(self, 'att'):
            self.att.reset_parameters()

    def forward(self, xs):
        r"""
        xs shape: [batch, nodes, featdim * num_layers]
        We split along the last dimension into separate layers, then run LSTM + attention.
        """
        # Split into list of shape: [batch, nodes, featdim]
        xs = torch.split(xs, self.channel, -1)
        # Stack -> [batch, nodes, num_layers, num_channels]
        xs = torch.stack(xs, 2)
        shape = xs.shape  # (B, N, L, C)
        # Reshape to (B*N, L, C)
        x = xs.reshape((-1, shape[2], shape[3]))
        alpha, _ = self.lstm(x)   # alpha: (B*N, L, 2*C') if bidirectional
        alpha = self.att(alpha).squeeze(-1)  # (B*N, L)
        alpha = torch.softmax(alpha, dim=-1)
        x = (x * alpha.unsqueeze(-1)).sum(dim=1)
        x = x.reshape((shape[0], shape[1], shape[3]))
        return x

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.mode)

########################################################
# GNN Module
########################################################

class GNN_Module(nn.Module):
    r"""
    A 3-layer block that either uses:
    - Our new DilatedDenseSAGEConv or DilatedDenseGINConv
    - Followed by BN + activation + optional linear
    - Optionally incorporates Jumping Knowledge
    """
    def __init__(self, input_dim, hidden_dim, embedding_dim, bias, bn, add_loop,
                 lin=True, gcn_name='SAGE', sync=False, activation='relu',
                 jk=False,
                 # Dilation parameters
                 T=10, k=2, layer_idx_start=1, dilation_mode='linear'
                 ):
        super(GNN_Module, self).__init__()
        self.jk = jk
        self.add_loop = add_loop
        self.activation_name = activation
        # We instantiate each layer, possibly with our dilated version
        self.gcn1 = self._gcn(gcn_name, input_dim, hidden_dim, bias,
                              T, k, layer_idx_start,   dilation_mode)
        self.active1 = self._activation(activation)
        self.gcn2 = self._gcn(gcn_name, hidden_dim, hidden_dim, bias,
                              T, k, layer_idx_start+1, dilation_mode)
        self.active2 = self._activation(activation)
        self.gcn3 = self._gcn(gcn_name, hidden_dim, embedding_dim, bias,
                              T, k, layer_idx_start+2, dilation_mode)
        self.active3 = self._activation(activation)

        self.bn_flag = bn
        if bn:
            if sync:
                self.bn1 = nn.SyncBatchNorm(hidden_dim)
                self.bn2 = nn.SyncBatchNorm(hidden_dim)
                self.bn3 = nn.SyncBatchNorm(embedding_dim)
            else:
                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim)
                self.bn3 = nn.BatchNorm1d(embedding_dim)

        if lin is True:
            self.lin = nn.Linear(2 * hidden_dim + embedding_dim, embedding_dim)
        else:
            self.lin = None

    def _activation(self, name='relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def _gcn(self, name, input_dim, hidden_dim, bias,
             T, k, layer_idx, dilation_mode):
        r"""
        Build either a dilated SAGE or GIN.
        """
        if name == 'SAGE':
            # Our Dilated version
            return DilatedDenseSAGEConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                bias=bias,
                T=T,
                k=k,
                layer_idx=layer_idx,
                dilation_mode=dilation_mode,
                normalize=True
            )
        else:
            # For GIN, we define an MLP then pass to DilatedDenseGINConv
            nn_mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                self._activation(self.activation_name),
                nn.Linear(hidden_dim, hidden_dim)
            )
            return DilatedDenseGINConv(
                nn=nn_mlp,
                T=T,
                k=k,
                layer_idx=layer_idx,
                dilation_mode=dilation_mode
            )

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, 'bn{}'.format(i))(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        # 1st layer
        x1 = self.gcn1(x, adj, mask, self.add_loop)
        x1 = self.active1(x1)
        if self.bn_flag:
            x1 = self.bn(1, x1)

        # 2nd layer
        x2 = self.gcn2(x1, adj, mask, self.add_loop)
        x2 = self.active2(x2)
        if self.bn_flag:
            x2 = self.bn(2, x2)

        # 3rd layer
        x3 = self.gcn3(x2, adj, mask, self.add_loop)
        x3 = self.active3(x3)
        if self.bn_flag:
            x3 = self.bn(3, x3)

        # Concatenate
        x_cat = torch.cat([x1, x2, x3], dim=-1)

        if mask is not None:
            x_cat = x_cat * mask

        if self.lin is not None:
            x_cat = self.lin(x_cat)
            if mask is not None:
                x_cat = x_cat * mask

        return x_cat

########################################################
# Full SoftPooling GCN with DiffPool
########################################################

class SoftPoolingGcnEncoder(nn.Module):
    r"""
    A hierarchical (DiffPool-like) model that uses the above GNN_Module blocks,
    now optionally with dilated neighbor skipping.
    """
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim,
                 bias, bn, assign_hidden_dim, label_dim,
                 assign_ratio=0.25, pred_hidden_dims=[50], concat=True,
                 gcn_name='SAGE', collect_assign=False, load_data_sparse=False,
                 norm_adj=False, activation='relu', drop_out=0.,
                 jk=False,
                 # Dilation parameters
                 T=10, k=2, dilation_mode='linear'
                 ):
        super(SoftPoolingGcnEncoder, self).__init__()
        self.jk = jk
        self.drop_out = drop_out
        self.norm_adj = norm_adj
        self.load_data_sparse = load_data_sparse
        self.collect_assign = collect_assign
        self.assign_matrix = []

        assign_dim = int(max_num_nodes * assign_ratio)

        # Stage 1
        self.GCN_embed_1 = GNN_Module(
            input_dim, hidden_dim, embedding_dim,
            bias, bn, add_loop=False, lin=False,
            gcn_name=gcn_name, activation=activation, jk=jk,
            T=T, k=k, layer_idx_start=1, dilation_mode=dilation_mode
        )
        if jk:
            self.jk1 = DenseJK('lstm', hidden_dim, 3)
        self.GCN_pool_1 = GNN_Module(
            input_dim, assign_hidden_dim, assign_dim,
            bias, bn, add_loop=False, gcn_name=gcn_name,
            activation=activation, jk=jk,
            T=T, k=k, layer_idx_start=1, dilation_mode=dilation_mode
        )

        if concat and not jk:
            input_dim_second = hidden_dim * 2 + embedding_dim
        else:
            input_dim_second = embedding_dim

        # Stage 2
        assign_dim = int(assign_dim * assign_ratio)
        self.GCN_embed_2 = GNN_Module(
            input_dim_second, hidden_dim, embedding_dim,
            bias, bn, add_loop=False, lin=False,
            gcn_name=gcn_name, activation=activation, jk=jk,
            T=T, k=k, layer_idx_start=4, dilation_mode=dilation_mode
        )
        if jk:
            self.jk2 = DenseJK('lstm', hidden_dim, 3)
        self.GCN_pool_2 = GNN_Module(
            input_dim_second, assign_hidden_dim, assign_dim,
            bias, bn, add_loop=False, gcn_name=gcn_name,
            activation=activation, jk=jk,
            T=T, k=k, layer_idx_start=4, dilation_mode=dilation_mode
        )

        # Stage 3
        self.GCN_embed_3 = GNN_Module(
            input_dim_second, hidden_dim, embedding_dim,
            bias, bn, add_loop=False, lin=False,
            gcn_name=gcn_name, activation=activation, jk=jk,
            T=T, k=k, layer_idx_start=7, dilation_mode=dilation_mode
        )
        if jk:
            self.jk3 = DenseJK('lstm', hidden_dim, 3)

        # Final MLP
        pred_input = input_dim_second * 3
        self.pred_model = self.build_readout_module(
            pred_input, pred_hidden_dims, label_dim, activation
        )

    @staticmethod
    def construct_mask(max_nodes, batch_num_nodes):
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def _re_norm_adj(self, adj, p, mask=None):
        """
        Re-normalize adjacency by setting diag= p, scaling off-diag to 1-p.
        """
        idx = torch.arange(0, adj.shape[1], device=adj.device)
        adj[:, idx, idx] = 0
        new_adj = torch.div(adj, adj.sum(-1)[..., None] + EPS) * (1 - p)
        new_adj[:, idx, idx] = p
        if mask is not None:
            new_adj = new_adj * mask
        return new_adj

    def _diff_pool(self, x, adj, s, mask):
        """
        Differentiable pooling with assignment matrix s.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        s = s.unsqueeze(0) if s.dim() == 2 else s
        batch_size, num_nodes, _ = x.size()
        s = torch.softmax(s, dim=-1)
        if self.collect_assign:
            self.assign_matrix.append(s.detach())
        if mask is not None:
            mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
            s = s * mask
        out = torch.matmul(s.transpose(1, 2), x)
        out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)
        return out, out_adj

    def _activation(self, name='relu'):
        assert name in ['relu', 'elu', 'leakyrelu']
        if name == 'relu':
            return nn.ReLU(inplace=True)
        elif name == 'elu':
            return nn.ELU(inplace=True)
        elif name == 'leakyrelu':
            return nn.LeakyReLU(inplace=True)

    def build_readout_module(self, pred_input_dim, pred_hidden_dims, label_dim, activation):
        """
        Build final MLP for classification.
        """
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self._activation(activation))
                pred_input_dim = pred_dim
                if self.drop_out > 0:
                    pred_layers.append(nn.Dropout(self.drop_out))
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def _sparse_to_dense_input(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        label = data.y
        edge_index = to_dense_adj(edge_index, batch)
        x, batch_num_node = to_dense_batch(x, batch)
        return x, edge_index, batch_num_node, label

    def forward(self, data):
        self.assign_matrix = []
        if self.load_data_sparse:
            x, adj, batch_num_nodes, label = self._sparse_to_dense_input(data)
        else:
            x, adj, batch_num_nodes = data[0], data[1], data[2]
            if self.training:
                label = data[3]

        max_num_nodes = adj.size()[1]
        embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)

        # Optionally renormalize adjacency
        if self.norm_adj:
            adj = self._re_norm_adj(adj, 0.4, embedding_mask)

        out_all = []

        # ------------------
        # Stage 1
        embed_feature = self.GCN_embed_1(x, adj, embedding_mask)
        if self.jk:
            embed_feature = self.jk1(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        assign = self.GCN_pool_1(x, adj, embedding_mask)
        x_pool, adj_pool = self._diff_pool(embed_feature, adj, assign, embedding_mask)

        # ------------------
        # Stage 2
        if self.norm_adj:
            adj_pool = self._re_norm_adj(adj_pool, 0.4)
        embed_feature = self.GCN_embed_2(x_pool, adj_pool, None)
        if self.jk:
            embed_feature = self.jk2(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        assign = self.GCN_pool_2(x_pool, adj_pool, None)
        x_pool, adj_pool = self._diff_pool(embed_feature, adj_pool, assign, None)

        # ------------------
        # Stage 3
        if self.norm_adj:
            adj_pool = self._re_norm_adj(adj_pool, 0.4)
        embed_feature = self.GCN_embed_3(x_pool, adj_pool, None)
        if self.jk:
            embed_feature = self.jk3(embed_feature)
        out, _ = torch.max(embed_feature, dim=1)
        out_all.append(out)
        output = torch.cat(out_all, 1)
        output = self.pred_model(output)
        if self.training:
            cls_loss = F.cross_entropy(output, label)
            return output, cls_loss
        return output

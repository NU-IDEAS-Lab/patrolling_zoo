import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import GCNConv, TransformerConv
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer

"""GNN modules."""

class GNNBase(nn.Module):
    ''' Base GNN module. '''

    def __init__(self, layers, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args):
        super(GNNBase, self).__init__()

        self.activation = [nn.Tanh(), nn.ReLU()][args.use_ReLU]

        layerSizes = []
        for i in range(layers, -1, -1):
            layerSizes.append(int(output_dim * (3 / 2) ** i))

        self.embedLayer = GNNLayer(node_dim, edge_dim, layerSizes[0], phi_dim, args)
        self.convLayer = nn.ModuleList()

        for i in range(layers):
            self.convLayer.append(GCNConv(
                layerSizes[i],
                layerSizes[i + 1],
                add_self_loops=False
            ))
            # self.convLayer.append(TransformerConv(
            #     layerSizes[i],
            #     layerSizes[i + 1],
            #     heads=3,
            #     concat=False,
            #     beta=False,
            #     dropout=0.0,
            #     edge_dim=edge_dim,
            #     bias=True,
            #     root_weight=True
            # ))

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, node_index=None) -> torch.Tensor:
        # Perform embedding
        x = self.embedLayer(x, edge_attr, edge_index)

        # Perform convolution.
        for layer in self.convLayer:
            x = layer(x, edge_index, edge_weight=edge_attr)
            # x = layer(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
        
        return x
    

    def gatherNodeFeats(self, x: torch.Tensor, idx: torch.Tensor):
        """
        This method is borrowed from InforMARL: https://github.com/nsidn98/InforMARL/blob/main/onpolicy/algorithms/utils/gnn.py#L346

        The output obtained from the network is of shape
        [batch_size, num_nodes, out_channels]. If we want to
        pull the features according to particular nodes in the
        graph as determined by the `idx`, use this
        Refer below link for more info on `gather()` method for 3D tensors
        https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4

        Args:
            x (Tensor): Tensor of shape (batch_size, num_nodes, out_channels)
            idx (Tensor): Tensor of shape (batch_size) or (batch_size, k)
                indicating the indices of nodes to pull from the graph

        Returns:
            Tensor: Tensor of shape (batch_size, out_channels) which just
                contains the features from the node of interest
        """
        out = []
        batch_size, num_nodes, num_feats = x.shape
        idx = idx.long()
        for i in range(idx.shape[1]):
            idx_tmp = idx[:, i].unsqueeze(-1)  # (batch_size, 1)
            assert idx_tmp.shape == (batch_size, 1)
            idx_tmp = idx_tmp.repeat(1, num_feats)  # (batch_size, out_channels)
            idx_tmp = idx_tmp.unsqueeze(1)  # (batch_size, 1, out_channels)
            gathered_node = x.gather(1, idx_tmp).squeeze(
                1
            )  # (batch_size, out_channels)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)  # (batch_size, out_channels*k)
        # out = out.squeeze(1)    # (batch_size, out_channels*k)

        return out


class GNNLayer(MessagePassing):
    ''' GNN layer with attentional aggregation.
        This class is based on two sources:
            * https://medium.com/the-modern-scientist/graph-neural-networks-series-part-4-the-gnns-message-passing-over-smoothing-e77ffee523cc
            * https://github.com/MIT-REALM/gcbf-pytorch/blob/main/gcbf/nn/gnn.py (Primarily)
    '''

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args):

        super(GNNLayer, self).__init__(
            # aggr=AttentionalAggregation(
            #     gate_nn=MLPLayer(
            #         input_dim=phi_dim,
            #         output_dim=1,
            #         hidden_size=128,
            #         layer_N=3,
            #         use_orthogonal=args.use_orthogonal,
            #         use_ReLU=args.use_ReLU,
            #         use_layer_norm=False,
            #         gain=1.0
            #     ),
            # )
            aggr="sum"
        )
        self.phi = MLPLayer(input_dim=2 * node_dim + edge_dim, output_dim=phi_dim, hidden_size=1024, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False, gain=1.0)
        self.gamma = MLPLayer(input_dim=phi_dim + node_dim, output_dim=output_dim, hidden_size=1024, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False, gain=1.0)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        res = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return res

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor = None, edge_attr: torch.Tensor = None) -> torch.Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)
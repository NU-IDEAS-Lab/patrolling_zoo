import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import AttentionalAggregation
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer

"""GNN modules."""


class GNNBase(MessagePassing):
    ''' GNN layer with attentional aggregation.
        This class is based on two sources:
            * https://medium.com/the-modern-scientist/graph-neural-networks-series-part-4-the-gnns-message-passing-over-smoothing-e77ffee523cc
            * https://github.com/MIT-REALM/gcbf-pytorch/blob/main/gcbf/nn/gnn.py (Primarily)
    '''

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args):

        print(f"Initialize GNNBase with node_dim={node_dim}, edge_dim={edge_dim}, output_dim={output_dim}, phi_dim={phi_dim}")

        super(GNNBase, self).__init__(aggr=AttentionalAggregation(
            gate_nn=MLPLayer(input_dim=phi_dim, output_dim=1, hidden_size=128, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU),))
        self.phi = MLPLayer(input_dim=2 * node_dim + edge_dim, output_dim=phi_dim, hidden_size=2048, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU)
        self.gamma = MLPLayer(input_dim=phi_dim + node_dim, output_dim=output_dim, hidden_size=2048, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor = None, edge_attr: torch.Tensor = None) -> torch.Tensor:
        info_ij = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.phi(info_ij)

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
        gamma_input = torch.cat([aggr_out, x], dim=1)
        return self.gamma(gamma_input)
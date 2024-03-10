import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import GraphSAGE

from onpolicy.algorithms.utils.gnn_conv import SAGEConvWithEdges

from typing import Tuple, Union, Final

"""GNN modules."""

class GNNBase(nn.Module):
    ''' Base GNN module. '''

    def __init__(self, layers: int, node_dim: int, edge_dim: int, output_dim: int, hidden_dim: int, node_type_idx: int,
                 node_type_dim: int = 1,
                 node_type_embed_dim: int = 2,
                 node_embedding_num: int = 2,
                 dropout_rate: float = 0.0,
                 jk = False,
                 **kwargs):
        super(GNNBase, self).__init__(**kwargs)

        self.node_type_idx = node_type_idx

        self.entity_embed = nn.Embedding(node_embedding_num, node_type_embed_dim)
        self.sage = GraphSAGEWithEdges(
            in_channels=node_dim - node_type_dim + node_type_embed_dim + edge_dim,
            hidden_channels=hidden_dim,
            out_channels=output_dim,
            num_layers=layers,
            dropout=dropout_rate,
            edge_channels=edge_dim,
            jk="cat" if jk else None,
            aggr="add"
        )


    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor, node_index=None) -> torch.Tensor:

        all_idx = torch.arange(0, x.shape[1])
        node_feat_idx = all_idx[all_idx != self.node_type_idx]
        
        # Extract node features and type for all nodes.
        node_feat = x[:, node_feat_idx]
        node_type = x[:, self.node_type_idx].int()
        node_type_embed = self.entity_embed(node_type)

        # Add dummy edge attributes for the first layer. In message passing, the edge attributes will be included as part of the node features.
        edge_feat = torch.zeros(node_feat.shape[0], edge_attr.shape[1], device=edge_attr.device)

        # Concatenate node features and type embeddings.
        info = torch.cat([node_feat, node_type_embed, edge_feat], dim=1)

        return self.sage(info, edge_index, edge_attr=edge_attr)
    

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
            gathered_node = x.gather(1, idx_tmp).squeeze(1)  # (batch_size, out_channels)
            out.append(gathered_node)
        out = torch.cat(out, dim=1)  # (batch_size, out_channels*k)

        return out


class GraphSAGEWithEdges(GraphSAGE):
    r"""All arguments the same as `torch_geometric.nn.GraphSAGE` except for the
    addition of `edge_attr` in the forward method.
    
    The Graph Neural Network from the `"Inductive Representation Learning
    on Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, using the
    :class:`~torch_geometric.nn.SAGEConv` operator for message passing.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.SAGEConv`.
    """

    supports_edge_weight: Final[bool] = False # we just consider weight an edge attribute...
    supports_edge_attr: Final[bool] = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        
        return SAGEConvWithEdges(in_channels, out_channels, **kwargs)
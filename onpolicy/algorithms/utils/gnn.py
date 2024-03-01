import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import GraphSAGE, SAGEConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn import GraphSAGE, SAGEConv
from torch_geometric.nn.dense.linear import Linear

from onpolicy.algorithms.utils.mlp import MLPLayer
from onpolicy.algorithms.utils.util import get_clones, init

from typing import Tuple, Union, Final, Optional, List
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, SparseTensor
from torch import Tensor
from torch_geometric.utils import spmm

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
            # root_weight=False,
            jk="cat" if jk else None,
            layer_count=layers
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
        # info = torch.cat([node_feat, node_type_embed], dim=1)

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
        # out = out.squeeze(1)    # (batch_size, out_channels*k)

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
        
        # This is some hacky stuff to avoid needing to rewrite the entire GraphSAGE class.
        # We concatenate the edge attributes to the node features in the message method.
        # Therefore, reduce the output size by the edge attribute size.
        i = len(self.convs)
        # if i < self.num_layers - 1:
        #     out_channels -= 2

        return SAGEConvWithEdges(in_channels, out_channels, idx=i, **kwargs)
    

class SAGEConvWithEdges(SAGEConv):
    r"""All arguments the same as `torch_geometric.nn.SAGEConv` except for the
    addition of `edge_attr` in the forward method.
    
    The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W}_2 \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    If :obj:`project = True`, then :math:`\mathbf{x}_j` will first get
    projected via

    .. math::
        \mathbf{x}_j \leftarrow \sigma ( \mathbf{W}_3 \mathbf{x}_j +
        \mathbf{b})

    as described in Eq. (3) of the paper.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        out_channels (int): Size of each output sample.
        aggr (str or Aggregation, optional): The aggregation scheme to use.
            Any aggregation of :obj:`torch_geometric.nn.aggr` can be used,
            *e.g.*, :obj:`"mean"`, :obj:`"max"`, or :obj:`"lstm"`.
            (default: :obj:`"mean"`)
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        project (bool, optional): If set to :obj:`True`, the layer will apply a
            linear transformation followed by an activation function before
            aggregation (as described in Eq. (3) of the paper).
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **inputs:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **outputs:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V_t}|, F_{out})` if bipartite
    """

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        *args,
        idx: int = 0,
        edge_channels: int = 0,
        layer_count: int = 1,
        **kwargs,
    ):

        super().__init__(
            in_channels,
            out_channels,
            *args,
            aggr=AttentionalAggregation(
                gate_nn=MLPLayer(
                    input_dim=in_channels,
                    output_dim=1,
                    hidden_size=128,
                    layer_N=2,
                    use_orthogonal=True,
                    use_ReLU=False,
                    use_layer_norm=False,
                    gain=nn.init.calculate_gain("tanh")
                ),
            ),
            **kwargs
        )
        self.edge_channels = edge_channels

        # self.gamma = MLPLayer(
        #     input_dim=self.in_channels + self.edge_channels,
        #     output_dim=self.in_channels,
        #     hidden_size=self.in_channels + self.edge_channels,
        #     layer_N=2,
        #     use_orthogonal=True,
        #     use_ReLU=True,
        #     use_layer_norm=True,
        #     gain=nn.init.calculate_gain("relu")
        # )
        self.idx = idx
        self.layer_count = layer_count
        # print(f"Created convolutional layer {idx} with input size {self.in_channels} and edge size {self.edge_channels}.")


    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None,
                edge_attr: OptTensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: OptTensor, edge_index: OptTensor = None) -> Tensor:
        # if self.idx == self.layer_count - 1:
        #     return torch.concatenate([x_j, edge_attr], dim=1)
        # else:
        #     return x_j

        # We assume that x_j has been expanded to include room for the edge attributes. This is done in GNNBase.
        x_j[:, -self.edge_channels:] = edge_attr
        return x_j

        info_ij = torch.concatenate([x_j, edge_attr], dim=1)
        return info_ij
    
    def update(self, aggr_out: Tensor, x: torch.Tensor = None) -> Tensor:
        ''' Called after aggregation occurs. '''

        # if self.idx < self.layer_count - 1:
        # if self.idx == 0:
        #     aggr_out[:, -1] = 0.0

        return aggr_out

    # def update(self, aggr_out: Tensor, x: torch.Tensor = None) -> Tensor:
    #     ''' Use a linear layer to restore the correct shape of the output. '''

    #     if aggr_out.shape[1] == self.in_channels:
    #         return aggr_out
    #     elif aggr_out.shape[1] == self.in_channels + self.edge_channels:
    #         return self.gamma(aggr_out)
    #     else:
    #         raise ValueError(f"Unexpected shape {aggr_out.shape} for aggr_out in SAGEConvWithEdges.update. Expected {self.in_channels} or {self.in_channels + self.edge_channels}.")

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        
        raise NotImplementedError("message_and_aggregate is not implemented for SAGEConvWithEdges")

        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
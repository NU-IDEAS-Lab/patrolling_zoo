import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn import GCNConv, TransformerConv, GraphSAGE, SAGEConv
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer

from onpolicy.algorithms.utils.util import get_clones, init
from typing import List, Tuple, Union, Optional, Final
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size, SparseTensor
from torch_geometric.utils import add_self_loops
from torch import Tensor
from torch_geometric.utils import spmm

"""GNN modules."""

class GNNBase(nn.Module):
    ''' Base GNN module. '''

    def __init__(self, layers, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args, node_type_idx):
        super(GNNBase, self).__init__()

        self.node_type_idx = node_type_idx

        self.entity_embed = nn.Embedding(2, 2)
        self.sage = GraphSAGEWithEdges(
            (
                node_dim - 1 + 2 + 2, #node_dim - nodetype + node_embed + edge_embed
                node_dim - 1 + 2
            ),
            output_dim,
            num_layers=1,
            dropout=0.5,
            # root_weight = False
        )


        return ########################################################################################################

        self.activation = [nn.Tanh(), nn.ReLU()][args.use_ReLU]

        layerSizes = []
        for i in range(layers, -1, -1):
            layerSizes.append(int(output_dim * (3 / 2) ** i))

        # self.embedLayer = GNNLayer(
        #     node_dim,
        #     edge_dim,
        #     layerSizes[0],
        #     phi_dim,
        #     args,
        #     node_type_idx=node_type_idx
        # )
        self.embedLayer = EmbedConv(
            node_dim - 1,
            num_embeddings=2,
            embedding_size=2,
            hidden_size=layerSizes[0],
            layer_N=1,
            use_orthogonal=args.use_orthogonal,
            use_ReLU=args.use_ReLU,
            use_layerNorm=False,
            add_self_loop=False,
            edge_dim=edge_dim,
            node_type_idx=node_type_idx
        )
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

        all_idx = torch.arange(0, x.shape[1])
        node_feat_idx = all_idx[all_idx != self.node_type_idx]
        
        # Extract node features and entity type for all nodes.
        node_feat = x[:, node_feat_idx]
        entity_type = x[:, self.node_type_idx].int()
        entity_embed = self.entity_embed(entity_type)

        info = torch.cat([node_feat, entity_embed], dim=1)

        # edge_attrs = edge_attr[:, 1:]
        # edge_weight = edge_attr[:, 0]

        return self.sage(info, edge_index, edge_attr=edge_attr)
        # return self.sage(info, edge_index, edge_weight=edge_weight, edge_attr=edge_attrs)

        return ########################################################################################################

        # Perform embedding
        # x = self.embedLayer(x, edge_attr, edge_index)
        x = self.embedLayer(x, edge_index, edge_attr=edge_attr)

        edge_weight = edge_attr[:, 0]

        # Perform convolution.
        for layer in self.convLayer:
            x = layer(x, edge_index, edge_weight=edge_weight)
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

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args, num_embeddings=2, embedding_size=2, node_type_idx=None):

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
            aggr="add"
        )

        gain = nn.init.calculate_gain(["tanh", "relu"][args.use_ReLU])
        self.node_type_idx = node_type_idx

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)

        phi_input_dim = 2 * (node_dim - 1 + embedding_size) + edge_dim
        self.phi = MLPLayer(input_dim=phi_input_dim, output_dim=output_dim, hidden_size=1024, layer_N=2, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False, gain=gain)
        # self.gamma = MLPLayer(input_dim=phi_dim + node_dim, output_dim=output_dim, hidden_size=1024, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False, gain=gain)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        res = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return res

    def message(self, x_j: torch.Tensor, x_i: torch.Tensor = None, edge_attr: torch.Tensor = None) -> torch.Tensor:
        ''' Message function for GNN layer.
            The entity type embedding in this function is based on that in
            https://github.com/nsidn98/InforMARL/blob/main/onpolicy/algorithms/utils/gnn.py '''

        all_idx = torch.arange(0, x_j.shape[1])
        node_feat_idx = all_idx[all_idx != self.node_type_idx]
        
        # Extract node features and entity type for node j.
        node_feat_j = x_j[:, node_feat_idx]
        entity_type_j = x_j[:, self.node_type_idx].int()
        entity_embed_j = self.entity_embed(entity_type_j)

        # Extract node features and entity type for node i.
        node_feat_i = x_i[:, node_feat_idx]
        entity_type_i = x_i[:, self.node_type_idx].int()
        entity_embed_i = self.entity_embed(entity_type_i)

        info_ij = torch.cat([node_feat_i, entity_embed_i, node_feat_j, entity_embed_j, edge_attr], dim=1)
        return self.phi(info_ij)


    # def update(self, aggr_out: torch.Tensor, x: torch.Tensor = None) -> torch.Tensor:
    #     gamma_input = torch.cat([aggr_out, x], dim=1)
    #     return self.gamma(gamma_input)


class EmbedConv(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_size: int,
        hidden_size: int,
        layer_N: int,
        use_orthogonal: bool,
        use_ReLU: bool,
        use_layerNorm: bool,
        add_self_loop: bool,
        edge_dim: int = 0,
        node_type_idx = None
    ):
        """
            NOTE: Adding this class from https://github.com/nsidn98/InforMARL/blob/main/onpolicy/algorithms/utils/gnn.py
            THIS IS A TEST

            EmbedConv Layer which takes in node features, node_type (entity type)
            and the  edge features (if they exist)
            `entity_embedding` is concatenated with `node_features` and
            `edge_features` and are passed through linear layers.
            The `message_passing` is similar to GCN layer

        Args:
            input_dim (int):
                The node feature dimension
            num_embeddings (int):
                The number of embedding classes aka the number of entity types
            embedding_size (int):
                The embedding layer output size
            hidden_size (int):
                Hidden layer size of the linear layers
            layer_N (int):
                Number of linear layers for aggregation
            use_orthogonal (bool):
                Whether to use orthogonal initialization for each layer
            use_ReLU (bool):
                Whether to use reLU for each layer
            use_layerNorm (bool):
                Whether to use layerNorm for each layer
            add_self_loop (bool):
                Whether to add self loops in the graph
            edge_dim (int, optional):
                Edge feature dimension, If zero then edge features are not
                considered. Defaults to 0.
        """
        super(EmbedConv, self).__init__(
            aggr="add"
            # aggr=AttentionalAggregation(
            #     gate_nn=MLPLayer(
            #         input_dim=128,
            #         output_dim=10,
            #         hidden_size=128,
            #         layer_N=3,
            #         use_orthogonal=use_orthogonal,
            #         use_ReLU=use_ReLU,
            #         use_layer_norm=True,
            #         gain=nn.init.calculate_gain(["tanh", "relu"][use_ReLU])
            #     ),
            # )
        )
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])
        self.node_type_idx = node_type_idx

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.entity_embed = nn.Embedding(num_embeddings, embedding_size)
        self.lin1 = nn.Sequential(
            init_(nn.Linear(input_dim + embedding_size + edge_dim, hidden_size)),
            active_func,
            layer_norm,
        )
        self.lin_h = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)), active_func, layer_norm
        )

        self.lin2 = get_clones(self.lin_h, self._layer_N)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
    ):
        if self._add_self_loops and edge_attr is None:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        all_idx = torch.arange(0, x_j.shape[1])
        node_feat_idx = all_idx[all_idx != self.node_type_idx]
        node_feat_j = x_j[:, node_feat_idx]
        # dont forget to convert to torch.LongTensor
        entity_type_j = x_j[:, self.node_type_idx].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x


class GraphSAGEWithEdges(GraphSAGE):
    r"""The Graph Neural Network from the `"Inductive Representation Learning
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

    supports_edge_weight: Final[bool] = False # we just consider it an edge attribute...
    supports_edge_attr: Final[bool] = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConvWithEdges(in_channels, out_channels, **kwargs)
    

class SAGEConvWithEdges(SAGEConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
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
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)


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

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        info_ij = torch.concatenate([x_j, edge_attr], dim=1)
        return info_ij

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: OptPairTensor) -> Tensor:
        
        raise NotImplementedError("message_and_aggregate is not implemented for SAGEConvWithEdges")

        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)
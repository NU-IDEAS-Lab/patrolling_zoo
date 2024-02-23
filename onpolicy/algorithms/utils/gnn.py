import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn import GCNConv, TransformerConv
import torch.nn as nn

from onpolicy.algorithms.utils.mlp import MLPLayer

from onpolicy.algorithms.utils.util import get_clones, init
from typing import List, Tuple, Union, Optional
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils import add_self_loops
from torch import Tensor

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
        # self.embedLayer = EmbedConv(
        #     node_dim - 1,
        #     num_embeddings=2,
        #     embedding_size=2,
        #     hidden_size=layerSizes[0],
        #     layer_N=1,
        #     use_orthogonal=args.use_orthogonal,
        #     use_ReLU=args.use_ReLU,
        #     use_layerNorm=False,
        #     add_self_loop=False,
        #     edge_dim=edge_dim
        # )
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
        # x = self.embedLayer(x, edge_index, edge_attr=edge_attr)

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

    def __init__(self, node_dim: int, edge_dim: int, output_dim: int, phi_dim: int, args, num_embeddings=2, embedding_size=2):

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

        node_feat_idx = [0, 2, 3, 4]
        
        # Extract node features and entity type for node j.
        node_feat_j = x_j[:, node_feat_idx]
        entity_type_j = x_j[:, 1].int()
        entity_embed_j = self.entity_embed(entity_type_j)

        # Extract node features and entity type for node i.
        node_feat_i = x_i[:, node_feat_idx]
        entity_type_i = x_i[:, 1].int()
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
        super(EmbedConv, self).__init__(aggr="add")
        self._layer_N = layer_N
        self._add_self_loops = add_self_loop
        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        layer_norm = [nn.Identity(), nn.LayerNorm(hidden_size)][use_layerNorm]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(["tanh", "relu"][use_ReLU])

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
        node_feat_idx = [0, 2, 3, 4]
        node_feat_j = x_j[:, node_feat_idx]
        # dont forget to convert to torch.LongTensor
        entity_type_j = x_j[:, 1].long()
        entity_embed_j = self.entity_embed(entity_type_j)
        if edge_attr is not None:
            node_feat = torch.cat([node_feat_j, entity_embed_j, edge_attr], dim=1)
        else:
            node_feat = torch.cat([node_feat_j, entity_embed_j], dim=1)
        x = self.lin1(node_feat)
        for i in range(self._layer_N):
            x = self.lin2[i](x)
        return x
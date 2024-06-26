import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase, MLPLayer
from onpolicy.algorithms.utils.gnn import GNNBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space, get_graph_obs_space, strip_graph_obs_space, get_graph_obs_space_idx

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

import numpy as np


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_gnn = args.use_gnn_policy
        self._use_gnn_mlp = args.use_gnn_mlp_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.device = device
        self.MAX_NEIGHBORS = 15
        self.MAX_NODES = 50

        if self._use_gnn:
            # Split up the graph and non-graph space.
            obs_space_graph = get_graph_obs_space(obs_space)
            obs_space_nongraph = strip_graph_obs_space(obs_space)
            self.obs_space_graph_idx = get_graph_obs_space_idx(obs_space)

            self.base = GNNBase(
                layers=args.gnn_layer_N,
                node_dim=get_shape_from_obs_space(obs_space_graph.node_space)[0],
                edge_dim=get_shape_from_obs_space(obs_space_graph.edge_space)[0],
                hidden_dim=args.gnn_hidden_size,
                output_dim=args.gnn_hidden_size, #self.hidden_size,
                node_type_idx=obs_space_graph.node_type_idx,
                node_type_dim=1,
                node_type_embed_dim=2,
                node_embedding_num=args.gnn_node_embedding_num,
                dropout_rate=args.gnn_dropout_rate,
                jk=args.gnn_skip_connections
            )

            

            self.neighbor_scorer = MLPLayer(input_dim=args.gnn_hidden_size, output_dim=1, hidden_size=self.hidden_size, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU, use_layer_norm=False)
            input_dim = self.MAX_NEIGHBORS + get_shape_from_obs_space(obs_space_nongraph)[0]

            if self._use_gnn_mlp:
                self.mlp0 = MLPLayer(input_dim=input_dim, output_dim=self.hidden_size, hidden_size=self.hidden_size, layer_N=3, use_orthogonal=args.use_orthogonal, use_ReLU=args.use_ReLU)
                input_dim = self.hidden_size
        else:
            obs_shape = get_shape_from_obs_space(obs_space)
            base = CNNBase if len(obs_shape) == 3 else MLPBase
            self.base = base(args, obs_shape)
            input_dim = self.hidden_size

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(input_dim, self.hidden_size, self._recurrent_N, self._use_orthogonal)
            input_dim = self.hidden_size

        self.act = ACTLayer(action_space, input_dim, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self._use_gnn:
            # Split observation into graph and non-graph components.
            obs_graph = obs[:, self.obs_space_graph_idx]
            nonGraphIdx = [i for i in range(obs.shape[1]) if i != self.obs_space_graph_idx]
            obs_nongraph = obs[:, nonGraphIdx]
            obs_nongraph = check(obs_nongraph.astype(np.float32)).to(**self.tpdv)

            # Batch the graphs and pass through GNN.
            graphs = Batch.from_data_list(obs_graph).to(self.device, "x", "edge_attr", "edge_index")
            actor_features = self.base(graphs.x, graphs.edge_attr, graphs.edge_index)

            # Restore the original shape of [batch_size, num_nodes (including agents), num_feats] from [batch_size*num_nodes, num_feats]
            actor_features, _ = to_dense_batch(actor_features, graphs.batch.to(self.device))
        
            if hasattr(graphs, "neighbors"):
                # Pad actor_features to max neighbors.
                if self.MAX_NODES - actor_features.shape[1] > 0:
                    actor_features = F.pad(actor_features, (0, 0, 0, self.MAX_NODES - actor_features.shape[1]), mode='constant', value=0.0)

                neighbors_mask = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()
                # Extend the mask for the full feature size.
                neighbors_mask = neighbors_mask.unsqueeze(2).repeat(1, 1, actor_features.shape[-1])
                actor_features_masked = torch.where(neighbors_mask, actor_features, 0.0)
                scores = self.neighbor_scorer(actor_features_masked)
                
                # Shift the scores to the correct position.
                scores_shifted = torch.zeros((actor_features.shape[0], self.MAX_NEIGHBORS), **self.tpdv)
                for i in range(actor_features.shape[0]):
                    nbrs = check(np.array(graphs.neighbors[i])).to(**self.tpdv).int()
                    scores_shifted[i, :nbrs.shape[0]] = scores[i, nbrs, 0]

                actor_features = scores_shifted

            elif hasattr(graphs, "agent_idx"):
                agent_idx = torch.from_numpy(np.array(graphs.agent_idx)).reshape(-1, 1).to(self.device)
                actor_features = self.base.gatherNodeFeats(actor_features, agent_idx)

            # Concatenate the graph and non-graph features.
            actor_features = torch.cat([actor_features, obs_nongraph], dim=-1)

            if self._use_gnn_mlp:
                actor_features = self.mlp0(actor_features)
        else:
            obs = check(obs).to(**self.tpdv)
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        if self._use_gnn:
            # Split observation into graph and non-graph components.
            obs_graph = obs[:, self.obs_space_graph_idx]
            nonGraphIdx = [i for i in range(obs.shape[1]) if i != self.obs_space_graph_idx]
            obs_nongraph = obs[:, nonGraphIdx]
            obs_nongraph = check(obs_nongraph.astype(np.float32)).to(**self.tpdv)

            # Batch the graphs and pass through GNN.
            graphs = Batch.from_data_list(obs_graph).to(self.device, "x", "edge_attr", "edge_index")
            actor_features = self.base(graphs.x, graphs.edge_attr, graphs.edge_index)

            # Restore the original shape of [batch_size, num_agents, num_feats] from [batch_size*num_agents, num_feats]
            actor_features, _ = to_dense_batch(actor_features, graphs.batch.to(self.device))
        
            if hasattr(graphs, "neighbors"):
                # Pad actor_features to max neighbors.
                if self.MAX_NODES - actor_features.shape[1] > 0:
                    actor_features = F.pad(actor_features, (0, 0, 0, self.MAX_NODES - actor_features.shape[1]), mode='constant', value=0.0)

                neighbors_mask = check(np.array(graphs.neighbors_mask)).to(**self.tpdv).bool()
                # Extend the mask for the full feature size.
                neighbors_mask = neighbors_mask.unsqueeze(2).repeat(1, 1, actor_features.shape[-1])
                actor_features_masked = torch.where(neighbors_mask, actor_features, 0.0)
                scores = self.neighbor_scorer(actor_features_masked)
                
                # Shift the scores to the correct position.
                scores_shifted = torch.zeros((actor_features.shape[0], self.MAX_NEIGHBORS), **self.tpdv)
                for i in range(actor_features.shape[0]):
                    nbrs = check(np.array(graphs.neighbors[i])).to(**self.tpdv).int()
                    scores_shifted[i, :nbrs.shape[0]] = scores[i, nbrs, 0]

                actor_features = scores_shifted

            elif hasattr(graphs, "agent_idx"):
                agent_idx = torch.from_numpy(np.array(graphs.agent_idx)).reshape(-1, 1).to(self.device)
                actor_features = self.base.gatherNodeFeats(actor_features, agent_idx)

            # Concatenate the graph and non-graph features.
            actor_features = torch.cat([actor_features, obs_nongraph], dim=-1)

            if self._use_gnn_mlp:
                actor_features = self.mlp0(actor_features)
        else:
            obs = check(obs).to(**self.tpdv)
            actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_gnn = args.use_gnn_critic
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        if self._use_gnn:
            raise NotImplementedError("GNN not implemented for R_Critic")
        else:
            cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
            base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
            self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)
        values = self.v_out(critic_features)

        return values, rnn_states

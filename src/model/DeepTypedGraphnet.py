import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean
from typing import Dict, Optional, Callable



class FeedForwardBlock(nn.Module):
    """
    Multi-layer perceptron with optional layer norm and learned distribution correction
    on the last layer. Activation is applied on all layers except the last one. Multiple
    inputs are concatenated before being fed to the MLP.
    """
    def __init__(self, layer_sizes, activation, use_layer_norm=False, concatenate_axis=-1):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.concatenate_axis = concatenate_axis
        
        self.layers = nn.ModuleList([nn.Linear(in_features, out_features) 
                                     for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])])
        
        self.layernorm = nn.LayerNorm(layer_sizes[-1]) if use_layer_norm else None
    
    def concatenate_args(self, args, axis: int = -1):
        """Concatenates all positional and keyword arguments on the given axis."""
        combined_args = list(args)
        return torch.cat(combined_args, dim=axis)

    def forward(self, *args):
        x = self.concatenate_args(args=args, axis=self.concatenate_axis)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        if self.layernorm:
            x = self.layernorm(x)
        return x

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)  # Optional: add normalization

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.norm(out + residual)  # skip connection + normalization
        return out

class FeedForwardBlock_Res(nn.Module):
    """
    Multi-layer perceptron with optional layer norm and learned distribution correction
    on the last layer. Activation is applied on all layers except the last one. Multiple
    inputs are concatenated before being fed to the MLP.
    """
    def __init__(self, layer_sizes, activation, use_layer_norm=False, concatenate_axis=-1):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.use_layer_norm = use_layer_norm
        self.concatenate_axis = concatenate_axis
        
        self.layers = nn.ModuleList([ResidualMLPBlock(in_features, out_features) if in_features == out_features else nn.Linear(in_features, out_features)
                                     for in_features, out_features in zip(layer_sizes[:-1], layer_sizes[1:])])
        
        self.layernorm = nn.LayerNorm(layer_sizes[-1]) if use_layer_norm else None
    
    def concatenate_args(self, args, axis: int = -1):
        """Concatenates all positional and keyword arguments on the given axis."""
        combined_args = list(args)
        return torch.cat(combined_args, dim=axis)

    def forward(self, *args):
        x = self.concatenate_args(args=args, axis=self.concatenate_axis)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        if self.layernorm:
            x = self.layernorm(x)
        return x
        

class GraphNetwork(nn.Module):
    def __init__(
        self,
        update_edge_fn: Callable,
        update_node_fn: Callable,
        aggregate_edges_for_nodes_fn: Callable = scatter_mean,
        mode: str = 'encode'
    ):
        super().__init__()
        self.update_edge_fn = update_edge_fn
        self.update_node_fn = update_node_fn
        self.aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
        self.mode = mode

    def _update_edges(self, edge_idx, edge_features, node_features):
        """Updates edges using their corresponding functions."""
        senders, receivers = edge_idx[...,0], edge_idx[...,1]
        if len(node_features.shape) == 2:
            sender_features = node_features[senders]
        elif len(node_features.shape) == 3:
            sender_features = node_features[:, senders, :]
        else:
            print('Unexpected node_feature dimension {0}'.format(len(node_features.shape)))
            exit()
        if self.mode == 'decode':
            edge_features = self.update_edge_fn(sender_features, edge_features)
        else:
            if len(node_features.shape) == 2:
                receiver_features = node_features[receivers]
            elif len(node_features.shape) == 3:
                receiver_features = node_features[:, receivers, :]
            else:
                print('Unexpected node_feature dimension {0}'.format(len(node_features.shape)))
                exit()
            edge_features = self.update_edge_fn(sender_features, receiver_features, edge_features)
        return edge_features

    def _update_nodes(self, edge_idx, edge_features, node_features):
        """Updates nodes using aggregated edge messages."""
        receivers = edge_idx[...,1]
        if self.mode == 'decode':
            mean_edges = self.aggregate_edges_for_nodes_fn(edge_features, receivers, dim=-2)
            node_features = self.update_node_fn(mean_edges)
        elif self.mode == 'encode':
            mean_edges = self.aggregate_edges_for_nodes_fn(edge_features, receivers, dim=-2)
            if len(node_features.shape) == 3:
                node_features = node_features[..., :mean_edges.shape[-2], :]
            elif len(node_features.shape) == 2:
                node_features = node_features[:mean_edges.shape[-2], :]
            node_features = self.update_node_fn(node_features, mean_edges)
        else:
            mean_edges = self.aggregate_edges_for_nodes_fn(edge_features, receivers, dim=-2)
            node_features = self.update_node_fn(node_features, mean_edges)
        return node_features
    
    def forward(self, edge_idx, edge_features, node_features):
        new_edge_features = self._update_edges(edge_idx, edge_features, node_features)
        new_node_features = self._update_nodes(edge_idx, new_edge_features, node_features)
        
        # if self.mode == 'encode':
        #     new_edge_features += edge_features
        #     if len(node_features.shape) == 3:
        #         new_node_features += node_features[..., :new_node_features.shape[-2], :]
        #     elif len(node_features.shape) == 2:
        #         new_node_features += node_features[:new_node_features.shape[-2], :]
        # elif self.mode != 'decode':
        #     new_edge_features += edge_features
        #     new_node_features += node_features
        return new_edge_features, new_node_features
    
class GraphMapFeatures(nn.Module):
    def __init__(
        self,
        embed_edge_fn: Optional[Callable] = None,
        embed_node_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.embed_edge_fn = embed_edge_fn
        self.embed_node_fn = embed_node_fn

    def forward(self, edge_features, node_features):
        new_edge_features, new_node_features = edge_features, node_features
        if self.embed_edge_fn:
            new_edge_features = self.embed_edge_fn(edge_features)
        if self.embed_node_fn:
            new_node_features = self.embed_node_fn(node_features)
        return new_edge_features, new_node_features


class DeepTypedGraphNet(nn.Module):
    def __init__(
        self,
        input_dim_edge,
        input_dim_node,
        edge_latent_size: int,
        node_latent_size: int,
        message_passing_steps: int,
        embed_edges: bool = True,
        embed_nodes: bool = True,
        edge_output_size: Optional[int] = None,
        node_output_size: Optional[int] = None,
        activation: str = 'swish',
        mode: str = 'encode'
    ):
        super().__init__()

        self.activation = _get_activation_fn(activation)
        self.embed_edges = embed_edges
        self.embed_nodes = embed_nodes
        self.message_passing_steps = message_passing_steps
        self.mode = mode

        # Node and edge embedders
        self.embed_edge_fn, self.embed_node_fn = None, None
        if embed_edges:
            self.embed_edge_fn =  FeedForwardBlock(layer_sizes=[input_dim_edge, edge_latent_size, edge_latent_size], activation=self.activation, use_layer_norm=True)

        if embed_nodes:
            self.embed_node_fn =  FeedForwardBlock(layer_sizes=[input_dim_node, node_latent_size, node_latent_size], activation=self.activation, use_layer_norm=True)

        self.embed_network = GraphMapFeatures(self.embed_edge_fn, self.embed_node_fn)

        # Message Passing Processors
        self.process_networks = nn.ModuleList()
        for i in range(self.message_passing_steps):
            if self.mode == 'decode':
                cur_process_edge_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
                cur_process_node_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size, node_latent_size, node_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
            else:
                cur_process_edge_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
                cur_process_node_fn = FeedForwardBlock(
                            layer_sizes=[node_latent_size + edge_latent_size, node_latent_size, node_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
            self.process_networks.append(GraphNetwork(cur_process_edge_fn, cur_process_node_fn, mode=self.mode))

        # Output decoders
        self.edge_output_fn, self.node_output_fn = None, None
        if edge_output_size:
            self.edge_output_fn =  FeedForwardBlock(layer_sizes=[edge_latent_size, edge_latent_size, edge_output_size], activation=self.activation, use_layer_norm=False)

        if node_output_size:
            self.node_output_fn =  FeedForwardBlock(layer_sizes=[node_latent_size, node_latent_size, node_output_size], activation=self.activation, use_layer_norm=False)

        self.output_network = GraphMapFeatures(self.edge_output_fn, self.node_output_fn)


    def forward(self, edge_idx, edge_features, node_features):
        """Runs the entire GNN model."""
        edge_features, node_features = self.embed_network(edge_features, node_features)
        for process_network in self.process_networks:
            new_edge_features, new_node_features = process_network(edge_idx, edge_features, node_features)
            if self.mode == 'encode':
                new_edge_features += edge_features
                if len(node_features.shape) == 3:
                    new_node_features += node_features[..., :new_node_features.shape[-2], :]
                elif len(node_features.shape) == 2:
                    new_node_features += node_features[:new_node_features.shape[-2], :]
            elif self.mode != 'decode':
                new_edge_features += edge_features
                new_node_features += node_features
            edge_features = new_edge_features
            node_features = new_node_features
        return self.output_network(edge_features, node_features)
    
def _get_activation_fn(name):
    """Returns the activation function."""
    activations = {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
        "swish": lambda x: x * torch.sigmoid(x)
    }
    return activations.get(name, F.relu)

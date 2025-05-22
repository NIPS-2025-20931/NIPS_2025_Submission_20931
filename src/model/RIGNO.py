from utils import *
from DeepTypedGraphnet import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial import Delaunay
from torch_scatter import scatter_mean
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Callable



class Encoder(nn.Module):
    # Encode point cloud to graph
    def __init__(self, input_dim_edge, input_dim_node, hidden_dim):
        super().__init__()
        edge_latent_size=hidden_dim
        node_latent_size=hidden_dim
        node_output_size=hidden_dim
        self.activation = lambda x: x * torch.sigmoid(x)
        self.embed_edge_fn =  FeedForwardBlock(layer_sizes=[input_dim_edge, edge_latent_size, edge_latent_size], activation=self.activation, use_layer_norm=True)
        self.embed_node_fn =  FeedForwardBlock(layer_sizes=[input_dim_node, node_latent_size, node_latent_size], activation=self.activation, use_layer_norm=True)
        cur_process_edge_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
        cur_process_node_fn = FeedForwardBlock(
                            layer_sizes=[node_latent_size + edge_latent_size, node_latent_size, node_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
        self.process_network = GraphNetwork(cur_process_edge_fn, cur_process_node_fn, mode='encode')
        self.node_output_fn =  FeedForwardBlock(layer_sizes=[node_latent_size, node_latent_size, node_output_size], activation=self.activation, use_layer_norm=False)

    def forward(self, edge_idx, edge_features, node_features):
        edge_features = self.embed_edge_fn(edge_features)
        node_features = self.embed_node_fn(node_features)
        edge_latents, node_latents = self.process_network(edge_idx, edge_features, node_features)
        node_latents += node_features[..., :node_latents.shape[-2], :]
        node_latents = self.node_output_fn(node_latents)
        return edge_latents, node_latents, node_features
    

class Processor(nn.Module):
    def __init__(self, input_dim_edge, input_dim_node, hidden_dim):
        super().__init__()
        edge_latent_size=hidden_dim
        node_latent_size=hidden_dim
        node_output_size=hidden_dim
        self.message_passing_steps = 4
        self.activation = lambda x: x * torch.sigmoid(x)
        self.embed_edge_fn =  FeedForwardBlock(layer_sizes=[input_dim_edge, edge_latent_size, edge_latent_size], activation=self.activation, use_layer_norm=True)
        # Message Passing Processors
        self.process_networks = nn.ModuleList()
        for i in range(self.message_passing_steps):
            cur_process_edge_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
            cur_process_node_fn = FeedForwardBlock(
                            layer_sizes=[node_latent_size + edge_latent_size, node_latent_size, node_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
            self.process_networks.append(GraphNetwork(cur_process_edge_fn, cur_process_node_fn, mode='process'))

        self.node_output_fn =  FeedForwardBlock(layer_sizes=[node_latent_size, node_latent_size, node_output_size], activation=self.activation, use_layer_norm=False)


    def forward(self, edge_idx, edge_features, node_features):
        edge_features = self.embed_edge_fn(edge_features)
        for process_network in self.process_networks:
            new_edge_features, new_node_features = process_network(edge_idx, edge_features, node_features)
            edge_features += new_edge_features
            node_features += new_node_features
        node_features = self.node_output_fn(node_features)
        return edge_features, node_features

class Decoder(nn.Module):
    def __init__(self, input_dim_edge, input_dim_node, hidden_dim, output_dim_node):
        super().__init__()
        edge_latent_size=hidden_dim
        node_latent_size=hidden_dim
        self.activation = lambda x: x * torch.sigmoid(x)
        self.embed_edge_fn =  FeedForwardBlock(layer_sizes=[input_dim_edge, edge_latent_size, edge_latent_size], activation=self.activation, use_layer_norm=True)
        self.process_edge_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            # layer_sizes=[edge_latent_size + node_latent_size, edge_latent_size, edge_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
        self.process_node_fn = FeedForwardBlock(
                            layer_sizes=[edge_latent_size + node_latent_size, node_latent_size, node_latent_size], 
                            # layer_sizes=[edge_latent_size, node_latent_size, node_latent_size], 
                            activation=self.activation, 
                            use_layer_norm=True)
        self.node_output_fn = FeedForwardBlock(layer_sizes=[node_latent_size, node_latent_size, output_dim_node], activation=torch.sigmoid, use_layer_norm=False)

    def _update_edges(self, edge_idx, edge_features, node_latents, node_features):
        """Updates edges using their corresponding functions."""
        senders, receivers = edge_idx[...,0], edge_idx[...,1]
        if len(node_latents.shape) == 2:
            sender_features = node_latents[senders]
            receiver_features = node_features[receivers]
        elif len(node_latents.shape) == 3:
            sender_features = node_latents[:, senders, :]
            receiver_features = node_features[:, receivers, :]
        else:
            print('Unexpected node_latents dimension {0}'.format(len(node_latents.shape)))
            print('Unexpected node_feature dimension {0}'.format(len(node_features.shape)))
            exit()
        edge_features = self.process_edge_fn(sender_features, receiver_features, edge_features)
        # edge_features = self.process_edge_fn(sender_features, edge_features)
        return edge_features
    
    def _update_nodes(self, edge_idx, edge_features, node_features):
        """Updates nodes using aggregated edge messages."""
        receivers = edge_idx[...,1]
        mean_edges = scatter_mean(edge_features, receivers, dim=-2)
        new_node_features = self.process_node_fn(node_features, mean_edges)
        # new_node_features = self.process_node_fn(mean_edges)
        return new_node_features

    def forward(self, edge_idx, edge_features, node_latents, node_features):
        edge_features = self.embed_edge_fn(edge_features)
        new_edge_features = self._update_edges(edge_idx, edge_features, node_latents, node_features)
        new_node_features = self._update_nodes(edge_idx, new_edge_features, node_features)
        output_features = self.node_output_fn(new_node_features)
        return edge_features, output_features


class RIGNO_SR(nn.Module):
    def __init__(self, input_dim_edge, input_dim_node, hidden_dim, output_dim_node):
        super().__init__()
        self.encoder = Encoder(input_dim_edge, input_dim_node, hidden_dim)
        self.processor = Processor(input_dim_edge, hidden_dim, hidden_dim)
        self.decoder = Decoder(input_dim_edge, hidden_dim, hidden_dim, output_dim_node)

    def forward(self, pc2g:Dict, g2g:Dict, g2pc:Dict):
        edge_idx, edge_features, node_features = pc2g['edge_idx'], pc2g['edge_features'], pc2g['node_features']
        edge_latents, node_latents, node_features = self.encoder(edge_idx, edge_features, node_features)

        edge_idx, edge_features = g2g['edge_idx'], g2g['edge_features']
        edge_latents, node_latents = self.processor(edge_idx, edge_features, node_latents)

        edge_idx, edge_features = g2pc['edge_idx'], g2pc['edge_features']
        edge_latents, node_features = self.decoder(edge_idx, edge_features, node_latents, node_features)

        return node_features

def PointCloudToGraph(points, nodes, radius):
    tree = cKDTree(points)
    edges = []
    
    for nodeidx in nodes:
        idxs = tree.query_ball_point(points[nodeidx], radius)
        edges += [[i, nodeidx] for i in idxs if i != nodeidx]
    
    return np.array(edges)

def GraphToPointCloud(nodes, query_points, radius):
    tree = cKDTree(nodes)
    edges = []

    nPoints = len(query_points)

    for pointidx in range(nPoints):
        idxs = tree.query_ball_point(query_points[pointidx], radius)
        edges += [[i, pointidx] for i in idxs]
    
    return np.array(edges)


def MultiResGraph(points, index, levels, pc2gfactor, g2gfactor, method):
    """
    Perform hierarchical sampling based on the given factor.
    
    Args:
        points (np.ndarray): The input array of shape (N, d).
        index (np.ndarray): The index array of shape (N,).
        factor (float): The sampling factor.

    Returns:
        list of tuples: Each tuple contains (sampled_data, reordered_index) at each level.
    """
    N, d = points.shape
    current_data = points
    current_index = index
    returned_data = np.copy(points)
    returned_index = np.copy(index)
    factor = pc2gfactor
    current_level = 0
    npts = []

    while N * factor >= 10 and current_level < levels:
        num_selected = int(N / factor)
        if method == 'random':
            selected_indices = np.random.choice(N, num_selected, replace=False)
        else:
            print('Method {0} is not defined'.format(method))
            exit()
        
        # Selected points and non-selected points
        selected_points = current_data[selected_indices]
        non_selected_indices = np.setdiff1d(np.arange(N), selected_indices)
        non_selected_points = current_data[non_selected_indices]

        # Reorder to maintain (N, d) shape
        reordered_data = np.vstack([selected_points, non_selected_points])
        reordered_index = np.hstack([current_index[selected_indices], current_index[non_selected_indices]])

        returned_data[:N] = reordered_data
        returned_index[:N] = reordered_index

        # Update for next iteration
        current_data = selected_points  # Use only the selected points for next iteration
        current_index = current_index[selected_indices]
        N = num_selected  # Update N
        factor = g2gfactor
        current_level += 1
        npts.append(num_selected)

    return returned_data, returned_index, npts


def GraphToGraphEdges(points, npts):
    all_edges = None
    for npt in npts:
        tri = Delaunay(points=points[:npt])
        indptr, cols = tri.vertex_neighbor_vertices
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        edges = np.stack([rows, cols], -1)
        edges = np.concatenate([edges, np.flip(edges, axis=-1)], axis=0)
        try:
            all_edges = np.vstack((all_edges, edges))
        except:
            all_edges = edges

    s = set()
    for e in all_edges:
        t = tuple(e)
        s.add(t)
    all_edges = np.array(list(s))
    return all_edges

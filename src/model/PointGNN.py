import torch.nn as nn
import torch.nn.functional as F
from utils import *
import torch
from torch_scatter import scatter_max
from torch_geometric.nn import global_max_pool
import time
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
import open3d

def multi_layer_downsampling(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False,):
    """Downsample the points using base_voxel_size at different scales"""
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    downsampled_list = [points_xyz]
    last_level = 0
    for level in levels:
        if np.isclose(last_level, level):
            downsampled_list.append(np.copy(downsampled_list[-1]))
        else:
            if add_rnd3d:
                xyz_idx = (points_xyz-xyz_offset+
                    base_voxel_size*level*np.random.random((1,3)))//\
                        (base_voxel_size*level)
                xyz_idx = xyz_idx.astype(np.int32)
                dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
                keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+\
                    xyz_idx[:, 2]*dim_y*dim_x
                sorted_order = np.argsort(keys)
                sorted_keys = keys[sorted_order]
                sorted_points_xyz = points_xyz[sorted_order]
                _, lens = np.unique(sorted_keys, return_counts=True)
                indices = np.hstack([[0], lens[:-1]]).cumsum()
                downsampled_xyz = np.add.reduceat(
                    sorted_points_xyz, indices, axis=0)/lens[:,np.newaxis]
                downsampled_list.append(np.array(downsampled_xyz))
            else:
                pcd = open3d.PointCloud()
                pcd.points = open3d.Vector3dVector(points_xyz)
                downsampled_xyz = np.asarray(open3d.voxel_down_sample(
                    pcd, voxel_size = base_voxel_size*level).points)
                downsampled_list.append(downsampled_xyz)
        last_level = level
    return downsampled_list

def multi_layer_downsampling_select(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales and match the downsampled
    points to original points by a nearest neighbor search.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.

    returns: vertex_coord_list, keypoint_indices_list
    """
    # Voxel downsampling
    vertex_coord_list = multi_layer_downsampling(
        points_xyz, base_voxel_size, levels=levels, add_rnd3d=add_rnd3d)
    num_levels = len(vertex_coord_list)
    assert num_levels == len(levels) + 1
    # Match downsampled vertices to original by a nearest neighbor search.
    keypoint_indices_list = []
    last_level = 0
    for i in range(1, num_levels):
        current_level = levels[i-1]
        base_points = vertex_coord_list[i-1]
        current_points = vertex_coord_list[i]
        if np.isclose(current_level, last_level):
            # same downsample scale (gnn layer),
            # just copy it, no need to search.
            vertex_coord_list[i] = base_points
            keypoint_indices_list.append(
                np.expand_dims(np.arange(base_points.shape[0]),axis=1))
        else:
            # different scale (pooling layer), search original points.
            nbrs = NearestNeighbors(n_neighbors=1,
                algorithm='kd_tree', n_jobs=1).fit(base_points)
            indices = nbrs.kneighbors(current_points, return_distance=False)
            vertex_coord_list[i] = base_points[indices[:, 0], :]
            keypoint_indices_list.append(indices)
        last_level = current_level
    return vertex_coord_list, keypoint_indices_list

def multi_layer_downsampling_random(points_xyz, base_voxel_size, levels=[1],
    add_rnd3d=False):
    """Downsample the points at different scales by randomly select a point
    within a voxel cell.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.

    returns: vertex_coord_list, keypoint_indices_list
    """
    xmax, ymax, zmax = np.amax(points_xyz, axis=0)
    xmin, ymin, zmin = np.amin(points_xyz, axis=0)
    xyz_offset = np.asarray([[xmin, ymin, zmin]])
    xyz_zeros = np.asarray([0, 0, 0], dtype=np.float32)
    vertex_coord_list = [points_xyz]
    keypoint_indices_list = []
    last_level = 0
    for level in levels:
        last_points_xyz = vertex_coord_list[-1]
        if np.isclose(last_level, level):
            # same downsample scale (gnn layer), just copy it
            vertex_coord_list.append(np.copy(last_points_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.arange(len(last_points_xyz)), axis=1))
        else:
            if not add_rnd3d:
                xyz_idx = (last_points_xyz - xyz_offset) \
                    // (base_voxel_size*level)
            else:
                xyz_idx = (last_points_xyz - xyz_offset +
                    base_voxel_size*level*np.random.random((1,3))) \
                        // (base_voxel_size*level)
            xyz_idx = xyz_idx.astype(np.int32)
            dim_x, dim_y, dim_z = np.amax(xyz_idx, axis=0) + 1
            keys = xyz_idx[:, 0]+xyz_idx[:, 1]*dim_x+xyz_idx[:, 2]*dim_y*dim_x
            num_points = xyz_idx.shape[0]

            voxels_idx = {}
            for pidx in range(len(last_points_xyz)):
                key = keys[pidx]
                if key in voxels_idx:
                    voxels_idx[key].append(pidx)
                else:
                    voxels_idx[key] = [pidx]

            downsampled_xyz = []
            downsampled_xyz_idx = []
            for key in voxels_idx:
                center_idx = random.choice(voxels_idx[key])
                downsampled_xyz.append(last_points_xyz[center_idx])
                downsampled_xyz_idx.append(center_idx)
            vertex_coord_list.append(np.array(downsampled_xyz))
            keypoint_indices_list.append(
                np.expand_dims(np.array(downsampled_xyz_idx),axis=1))
        last_level = level

    return vertex_coord_list, keypoint_indices_list

def gen_multi_level_local_graph_v3(
    points_xyz, base_voxel_size, level_configs, add_rnd3d=False,
    downsample_method='center'):
    """Generating graphs at multiple scale. This function enforce output
    vertices of a graph matches the input vertices of next graph so that
    gnn layers can be applied sequentially.

    Args:
        points_xyz: a [N, D] matrix. N is the total number of the points. D is
        the dimension of the coordinates.
        base_voxel_size: scalar, the cell size of voxel.
        level_configs: a dict of 'level', 'graph_gen_method',
        'graph_gen_kwargs', 'graph_scale'.
        add_rnd3d: boolean, whether to add random offset when downsampling.
        downsample_method: string, the name of downsampling method.
    returns: vertex_coord_list, keypoint_indices_list, edges_list
    """
    if isinstance(base_voxel_size, list):
        base_voxel_size = np.array(base_voxel_size)
    # Gather the downsample scale for each graph
    scales = [config['graph_scale'] for config in level_configs]
    # Generate vertex coordinates
    if downsample_method=='center':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_select(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    if downsample_method=='random':
        vertex_coord_list, keypoint_indices_list = \
            multi_layer_downsampling_random(
                points_xyz, base_voxel_size, scales, add_rnd3d=add_rnd3d)
    # Create edges
    edges_list = []
    for config in level_configs:
        graph_level = config['graph_level']
        gen_graph_fn = get_graph_generate_fn(config['graph_gen_method'])
        method_kwarg = config['graph_gen_kwargs']
        points_xyz = vertex_coord_list[graph_level]
        center_xyz = vertex_coord_list[graph_level+1]
        vertices = gen_graph_fn(points_xyz, center_xyz, **method_kwarg)
        edges_list.append(vertices)
    return vertex_coord_list, keypoint_indices_list, edges_list

def gen_disjointed_rnn_local_graph_v3(
    points_xyz, center_xyz, radius, num_neighbors,
    neighbors_downsample_method='random',
    scale=None):
    """Generate a local graph by radius neighbors.
    """
    if scale is not None:
        scale = np.array(scale)
        points_xyz = points_xyz/scale
        center_xyz = center_xyz/scale
    nbrs = NearestNeighbors(
        radius=radius,algorithm='ball_tree', n_jobs=1, ).fit(points_xyz)
    indices = nbrs.radius_neighbors(center_xyz, return_distance=False)
    if num_neighbors > 0:
        if neighbors_downsample_method == 'random':
            indices = [neighbors if neighbors.size <= num_neighbors else
                np.random.choice(neighbors, num_neighbors, replace=False)
                for neighbors in indices]
    vertices_v = np.concatenate(indices)
    vertices_i = np.concatenate(
        [i*np.ones(neighbors.size, dtype=np.int32)
            for i, neighbors in enumerate(indices)])
    vertices = np.array([vertices_v, vertices_i]).transpose()
    return vertices

def get_graph_generate_fn(method_name):
    method_map = {
        'disjointed_rnn_local_graph_v3':gen_disjointed_rnn_local_graph_v3,
        'multi_level_local_graph_v3': gen_multi_level_local_graph_v3,
    }
    return method_map[method_name]
   

class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        torch.nn.init.xavier_normal_(self.linear.weight)
        if bias:
            torch.nn.init.normal_(self.linear.bias, 0, 0.001)
        self.act = activation_fn
    
    def forward(self, x):
        if self.act is None:
            return self.linear(x)
        return self.act(self.linear(x))

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation=nn.ReLU(), bias=True, activate_final=False):
        super().__init__()
        self.net = []
        self.net.append(MLPLayer(input_dim, hidden_dim, activation, bias=bias))
        for i in range(num_layers-1):
            self.net.append(MLPLayer(hidden_dim, hidden_dim, activation, bias=bias))
            
        if activate_final:
            self.net.append(MLPLayer(hidden_dim, output_dim, activation, bias=bias))
        else:
            self.net.append(MLPLayer(hidden_dim, output_dim, None, bias=bias))
            
        self.net = nn.Sequential(*self.net)
        
    def forward(self, input):
        output = self.net(input)
        
        return output

def multi_layer_neural_network_fn(Ks):
    linears = []
    for i in range(1, len(Ks)):
        linears += [
        nn.Linear(Ks[i-1], Ks[i]),
        nn.ReLU(),
        nn.BatchNorm1d(Ks[i])]
    return nn.Sequential(*linears)

def multi_layer_fc_fn(Ks=[300, 64, 32, 64], num_classes=4, is_logits=False, num_layers=4):
    assert len(Ks) == num_layers
    linears = []
    for i in range(1, len(Ks)):
        linears += [
                nn.Linear(Ks[i-1], Ks[i]),
                nn.ReLU(),
                nn.BatchNorm1d(Ks[i])
                ]

    if is_logits:
        linears += [
                nn.Linear(Ks[-1], num_classes)]
    else:
        linears += [
                nn.Linear(Ks[-1], num_classes),
                nn.ReLU(),
                nn.BatchNorm1d(num_classes)
                ]
    return nn.Sequential(*linears)

def max_aggregation_fn(features, index, l):
    """
    Arg: features: N x dim
    index: N x 1, e.g.  [0,0,0,1,1,...l,l]
    l: lenght of keypoints

    """
    index = index.unsqueeze(-1).expand(-1, features.shape[-1]) # N x 64
    set_features = torch.zeros((l, features.shape[-1]), device=features.device).permute(1,0).contiguous() # len x 64
    set_features, argmax = scatter_max(features.permute(1,0), index.permute(1,0), out=set_features)
    set_features = set_features.permute(1,0)
    return set_features

def focal_loss_sigmoid(labels, logits, alpha=0.5, gamma=2):
     """
     github.com/tensorflow/models/blob/master/\
         research/object_detection/core/losses.py
     Computer focal loss for binary classification
     Args:
       labels: A int32 tensor of shape [batch_size]. N x 1
       logits: A float32 tensor of shape [batch_size]. N x C
       alpha: A scalar for focal loss alpha hyper-parameter.
       If positive samples number > negtive samples number,
       alpha < 0.5 and vice versa.
       gamma: A scalar for focal loss gamma hyper-parameter.
     Returns:
       A tensor of the same shape as `labels`
     """

     prob = logits.sigmoid()
     labels = torch.nn.functional.one_hot(labels.squeeze().long(), num_classes=prob.shape[1])

     cross_ent = torch.clamp(logits, min=0) - logits * labels + torch.log(1+torch.exp(-torch.abs(logits)))
     prob_t = (labels*prob) + (1-labels) * (1-prob)
     modulating = torch.pow(1-prob_t, gamma)
     alpha_weight = (labels*alpha)+(1-labels)*(1-alpha)

     focal_cross_entropy = modulating * alpha_weight * cross_ent
     return focal_cross_entropy

class PointSetPooling(nn.Module):
    def __init__(self, point_MLP_depth_list=[4, 32, 64, 128, 300], output_MLP_depth_list=[300, 300, 300]):
        super(PointSetPooling, self).__init__()

        Ks = list(point_MLP_depth_list)
        self.point_linears = multi_layer_neural_network_fn(Ks)
        
        Ks = list(output_MLP_depth_list)
        self.out_linears = multi_layer_neural_network_fn(Ks)

    def forward(self, 
            point_features,
            point_coordinates,
            keypoint_indices,
            set_indices):
        """apply a features extraction from point sets.
        Args:
            point_features: a [N, M] tensor. N is the number of points.
            M is the length of the features.
            point_coordinates: a [N, D] tensor. N is the number of points.
            D is the dimension of the coordinates.
            keypoint_indices: a [K, 1] tensor. Indices of K keypoints.
            set_indices: a [S, 2] tensor. S pairs of (point_index, set_index).
            i.e. (i, j) indicates point[i] belongs to the point set created by
            grouping around keypoint[j].
        returns: a [K, output_depth] tensor as the set feature.
        Output_depth depends on the feature extraction options that
        are selected.
        """

        #print(f"point_features: {point_features.shape}")
        #print(f"point_coordinates: {point_coordinates.shape}")
        #print(f"keypoint_indices: {keypoint_indices.shape}")
        #print(f"set_indices: {set_indices.shape}")

        # Gather the points in a set
        point_set_features = point_features[set_indices[:, 0]]
        point_set_coordinates = point_coordinates[set_indices[:, 0]]
        point_set_keypoint_indices = keypoint_indices[set_indices[:, 1]]

        #point_set_keypoint_coordinates_1 = point_features[point_set_keypoint_indices[:, 0]]
        point_set_keypoint_coordinates = point_coordinates[point_set_keypoint_indices[:, 0]]

        point_set_coordinates = point_set_coordinates - point_set_keypoint_coordinates
        point_set_features = torch.cat([point_set_features, point_set_coordinates], axis=-1)

        # Step 1: Extract all vertex_features
        extracted_features = self.point_linears(point_set_features) # N x 64

        # Step 2: Aggerate features using scatter max method.
        #index = set_indices[:, 1].unsqueeze(-1).expand(-1, extracted_features.shape[-1]) # N x 64
        #set_features = torch.zeros((len(keypoint_indices), extracted_features.shape[-1]), device=extracted_features.device).permute(1,0).contiguous() # len x 64
        #set_features, argmax = scatter_max(extracted_features.permute(1,0), index.permute(1,0), out=set_features)
        #set_features = set_features.permute(1,0)

        set_features = max_aggregation_fn(extracted_features, set_indices[:, 1], len(keypoint_indices))

        # Step 3: MLP for set_features
        set_features = self.out_linears(set_features)
        return set_features

class GraphNetAutoCenter(nn.Module):
    def __init__(self, auto_offset=True, auto_offset_MLP_depth_list=[300, 64, 3], edge_MLP_depth_list=[303, 300, 300], update_MLP_depth_list=[300, 300, 300]):
        super(GraphNetAutoCenter, self).__init__()
        self.auto_offset = auto_offset
        self.auto_offset_fn = multi_layer_neural_network_fn(auto_offset_MLP_depth_list)
        self.edge_feature_fn = multi_layer_neural_network_fn(edge_MLP_depth_list)
        self.update_fn = multi_layer_neural_network_fn(update_MLP_depth_list)


    def forward(self, input_vertex_features,
        input_vertex_coordinates,
        keypoint_indices,
        edges):
        """apply one layer graph network on a graph. .
        Args:
            input_vertex_features: a [N, M] tensor. N is the number of vertices.
            M is the length of the features.
            input_vertex_coordinates: a [N, D] tensor. N is the number of
            vertices. D is the dimension of the coordinates.
            NOT_USED: leave it here for API compatibility.
            edges: a [K, 2] tensor. K pairs of (src, dest) vertex indices.
        returns: a [N, M] tensor. Updated vertex features.
        """
        #print(f"input_vertex_features: {input_vertex_features.shape}")
        #print(f"input_vertex_coordinates: {input_vertex_coordinates.shape}")
        #print(NOT_USED)
        #print(f"edges: {edges.shape}")

        # Gather the source vertex of the edges
        s_vertex_features = input_vertex_features[edges[:, 0]]
        s_vertex_coordinates = input_vertex_coordinates[edges[:, 0]]

        if self.auto_offset:
            offset = self.auto_offset_fn(input_vertex_features)
            input_vertex_coordinates = input_vertex_coordinates + offset

        # Gather the destination vertex of the edges
        d_vertex_coordinates = input_vertex_coordinates[edges[:, 1]]

        # Prepare initial edge features
        edge_features = torch.cat([s_vertex_features, s_vertex_coordinates - d_vertex_coordinates], dim=-1)
        
        # Extract edge features
        edge_features = self.edge_feature_fn(edge_features)

        # Aggregate edge features
        aggregated_edge_features = max_aggregation_fn(edge_features, edges[:,1], len(keypoint_indices))

        # Update vertex features
        update_features = self.update_fn(aggregated_edge_features)
        output_vertex_features  = update_features + input_vertex_features
        return output_vertex_features 

class AggregationLayer(nn.Module):
    def __init__(self, out_features, activate_final=True):
        super().__init__()
        self.out_features = out_features
        self.mlp = MLP(input_dim=300, hidden_dim=300, output_dim=out_features, num_layers=2, bias=True, activate_final=True)
        
    def forward(self, features, edge_idx, device):
        predict = self.mlp(features)
        # max pooling
        latent = global_max_pool(predict, batch=None)
        return latent

class PointGNNEncoder(nn.Module):
    def __init__(self, out_features, activate_final=True, graph_net_layers=3):
        super(PointGNNEncoder, self).__init__()
        self.point_set_pooling = PointSetPooling()
        self.graph_nets = nn.ModuleList()
        for i in range(graph_net_layers):
            self.graph_nets.append(GraphNetAutoCenter())
        self.aggregation = AggregationLayer(out_features=out_features, activate_final=activate_final)

    def forward(self, batch, device):
        input_v, vertex_coord_list, keypoint_indices_list, edges_list = batch

        point_features, point_coordinates, keypoint_indices, set_indices = input_v, vertex_coord_list[0], keypoint_indices_list[0], edges_list[0]
        point_features = self.point_set_pooling(point_features, point_coordinates, keypoint_indices, set_indices)


        point_coordinates, keypoint_indices, set_indices = vertex_coord_list[1], keypoint_indices_list[1], edges_list[1]
        for i, graph_net in enumerate(self.graph_nets):
            point_features = graph_net(point_features, point_coordinates, keypoint_indices, set_indices)
        latent = self.aggregation(point_features, set_indices, device)
        return latent

class PointGNN_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, pc_features, graph_net_layers=3, pos_freq_const=10, pos_freq_num=30, enc_act_final=True):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = PointGNNEncoder(out_features=pc_features, activate_final=enc_act_final, graph_net_layers=graph_net_layers)
        self.pos_features = in_features*pos_freq_num*2
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + pc_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, 1, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, lowres_pos, lowres_feats, coords, config, device):
        assert lowres_pos.shape[0] == lowres_feats.shape[0]
        assert lowres_pos.shape[0] == coords.shape[0]
        
        batch_size = coords.shape[0]
        latents = None
        for b in range(batch_size):
            # Graph Generation
            graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
            (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(lowres_pos[b], **config['graph_gen_kwargs'])
            vertex_coord_list = [torch.from_numpy(p.astype(np.float32)) for p in vertex_coord_list]
            keypoint_indices_list = [torch.from_numpy(e.astype(np.int32)).long() for e in keypoint_indices_list]
            edges_list = [torch.from_numpy(e.astype(np.int32)).long() for e in edges_list]
            curr_batch = (lowres_feats[b].reshape(-1, 1), vertex_coord_list, keypoint_indices_list, edges_list)
            new_batch = []
            for item in curr_batch:
                if not isinstance(item, torch.Tensor):
                    item = [x.to(device) for x in item]
                else:
                    item = item.to(device)
                new_batch += [item]
            curr_batch = new_batch
            # Extract latent
            g_latent = self.lowres_encoding(curr_batch, device)
            pos_latent = self.pos_encoding(coords[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[0], 1)
            latent = torch.cat([pos_latent, g_latent], dim=-1)
            latent = latent[None, ...]
            if latents == None:
                latents = latent
            else:
                latents = torch.vstack((latents, latent))
        output = self.net(latents)
        return output
    

class PointGNN_SR_3D(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, pc_features, graph_net_layers=3, pos_freq_const=10, pos_freq_num=30, enc_act_final=True):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = PointGNNEncoder(out_features=pc_features, activate_final=enc_act_final, graph_net_layers=graph_net_layers)
        self.pos_features = in_features*pos_freq_num*3
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + pc_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, 1, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, lowres_pos, lowres_feats, coords, config, device):
        assert lowres_pos.shape[0] == lowres_feats.shape[0]
        assert lowres_pos.shape[0] == coords.shape[0]
        
        batch_size = coords.shape[0]
        latents = None
        for b in range(batch_size):
            # Graph Generation
            graph_generate_fn = get_graph_generate_fn(config['graph_gen_method'])
            (vertex_coord_list, keypoint_indices_list, edges_list) = graph_generate_fn(lowres_pos[b], **config['graph_gen_kwargs'])
            vertex_coord_list = [torch.from_numpy(p.astype(np.float32)) for p in vertex_coord_list]
            keypoint_indices_list = [torch.from_numpy(e.astype(np.int32)).long() for e in keypoint_indices_list]
            edges_list = [torch.from_numpy(e.astype(np.int32)).long() for e in edges_list]
            curr_batch = (lowres_feats[b].reshape(-1, 1), vertex_coord_list, keypoint_indices_list, edges_list)
            new_batch = []
            for item in curr_batch:
                if not isinstance(item, torch.Tensor):
                    item = [x.to(device) for x in item]
                else:
                    item = item.to(device)
                new_batch += [item]
            curr_batch = new_batch
            # Extract latent
            g_latent = self.lowres_encoding(curr_batch, device)
            pos_latent = self.pos_encoding(coords[b])
            g_latent = g_latent[0].repeat(pos_latent.shape[0], 1)
            latent = torch.cat([pos_latent, g_latent], dim=-1)
            latent = latent[None, ...]
            if latents == None:
                latents = latent
            else:
                latents = torch.vstack((latents, latent))
        output = self.net(latents)
        return output
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_max_pool
from utils import *

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)

def farthest_point_sample(xy, npoint):
    """
    Input:
        xy: pointcloud data, [B, N, 2]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xy.device
    B, N, C = xy.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xy[batch_indices, farthest, :].view(B, 1, 2)
        dist = torch.sum((xy - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xy, new_xy):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xy: all points, [B, N, 2]
        new_xy: query points, [B, S, 2]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xy.device
    B, N, C = xy.shape
    _, S, _ = new_xy.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xy, xy)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xy, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xy: input points position data, [B, N, 2]
        points: input points data, [B, N, D]
    Return:
        new_xy: sampled points position data, [B, npoint, nsample, 2]
        new_points: sampled points data, [B, npoint, nsample, 2+D]
    """
    B, N, C = xy.shape
    S = npoint
    fps_idx = farthest_point_sample(xy, npoint) # [B, npoint]
    torch.cuda.empty_cache()
    new_xy = index_points(xy, fps_idx)
    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xy, xy)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xy, new_xy)
    torch.cuda.empty_cache()
    grouped_xy = index_points(xy, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xy_norm = grouped_xy - new_xy.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xy_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xy_norm
    if returnfps:
        return new_xy, new_points, grouped_xy, fps_idx
    else:
        return new_xy, new_points


def sample_and_group_all(xy, points):
    """
    Input:
        xy: input points position data, [B, N, 2]
        points: input points data, [B, N, D]
    Return:
        new_xy: sampled points position data, [B, 1, 2]
        new_points: sampled points data, [B, 1, N, 2+D]
    """
    device = xy.device
    B, N, C = xy.shape
    new_xy = torch.zeros(B, 1, C).to(device)
    grouped_xy = xy.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xy, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xy
    return new_xy, new_points

################################################################################################################################################################################

class TransitionDown(nn.Module):
    def __init__(self, nneighbor, channels):
        super().__init__()
        self.radius = 0
        self.nsample = nneighbor
        in_channel = channels[0]
        mlp = channels[1:]
        last_channel = in_channel
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        
    def forward(self, xy, points):
        assert len(xy.shape) == 3
        new_xy, new_points = sample_and_group(xy.shape[1] // 4, self.radius, self.nsample, xy, points, knn=True)
        # new_xy: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))
        # [B, C+D, nsample, npoints] -> [B, C+D, npoints] -> [B, npoints, C+D]
        # maxpooling along nsample dimension
        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xy, new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xy: b x n x 2, features: b x n x f
    def forward(self, xy, features):
        dists = square_distance(xy, xy)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xy = index_points(xy, knn_idx)
        
        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xy[:, :, None] - knn_xy)  # b x n x k x f
        
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res, attn

class Backbone(nn.Module):
    def __init__(self, nblocks, nneighbor, feat_dim, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor)
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        for i in range(nblocks):
            channel = 32 * 2 ** (i + 1) # [64, 128, 256, 512]
            self.transition_downs.append(TransitionDown(nneighbor, [channel // 2 + 2, channel, channel]))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.nblocks = nblocks
    
    def forward(self, x):
        xy = x[..., :2]
        points = self.transformer1(xy, self.fc1(x))[0]

        xy_and_feats = [(xy, points)]
        for i in range(self.nblocks):
            xy, points = self.transition_downs[i](xy, points)
            points = self.transformers[i](xy, points)[0]
            xy_and_feats.append((xy, points))
        return points, xy_and_feats


class PointTransformerEncoder(nn.Module):
    def __init__(self, nblocks, nneighbor, latent_features, feat_dim, transformer_dim):
        super().__init__()
        self.backbone = Backbone(nblocks, nneighbor, feat_dim, transformer_dim)
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** nblocks, 256),
            nn.ReLU(),
            nn.Linear(256, latent_features)
        )
        self.nblocks = nblocks
    
    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        # res = self.fc2(points)
        # res = global_max_pool(res, batch=None)
        return res

################################################################################################################################################################################

class PT_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, enc_layers, knn_neighbors, latent_features, feat_dim, transformer_dim, pos_freq_const=10, pos_freq_num=30):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = PointTransformerEncoder(enc_layers, knn_neighbors, latent_features, feat_dim, transformer_dim)
        self.pos_features = in_features*pos_freq_num*2
        self.hidden_features = hidden_features
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent_features = latent_features
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, 1, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, pointcloud_feats):
        assert coords.shape[0] == pointcloud_feats.shape[0]
        pc_latent = self.lowres_encoding(pointcloud_feats)
        pos_latent = self.pos_encoding(coords)
        pc_latent = pc_latent.repeat(1, pos_latent.shape[1])
        pc_latent = pc_latent.reshape((coords.shape[0], pos_latent.shape[1], self.latent_features))
        latents = torch.cat([pos_latent, pc_latent], dim=-1)
        output = self.net(latents)
        return output
    
    def get_latents(self, pointcloud_feats):
        return self.lowres_encoding(pointcloud_feats)

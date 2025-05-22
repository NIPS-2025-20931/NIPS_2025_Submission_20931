import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import json
from utils import *


def entropy_sampling(xyz, points, nBins, npoint):
    """
    Input:
        xyz: input points position data, [B, C, N] (if 2D use xy0)
        points: input points data, [B, D, M] (pressure in this case)
        nBins: number of bins for histogram
        npoint: number of samples
    Return:
        centroids: sampled pointcloud points, [B, npoint]
    """
    sampled_xyz_batches = []
    sampled_feats_batches = []
    for bidx in range(points.shape[0]):
        # build histogram and sort by number of elements
        minval = torch.min(points[bidx][...,0])
        maxval = torch.max(points[bidx][...,0])
        hist = torch.histc(points[bidx][...,0], bins=nBins, min=minval, max=maxval)
        bin_min = minval
        bin_width = (maxval - minval) / nBins
        suffle_idx = torch.randperm(points[bidx].size()[0])
        shuffle_xyz = xyz[bidx][suffle_idx]
        shuffle_feat = points[bidx][suffle_idx]
        sort_hist = []
        for i in range(len(hist)):
            sort_hist.append(torch.Tensor([hist[i], minval + bin_width*i, minval + bin_width*(i+1)]))
        sort_hist = torch.stack(sort_hist)
        hist, ind = sort_hist.sort(0)
        sort_hist = sort_hist[ind[...,0]]

        # compute the number of points for each bin
        sampled_xyz = []
        sampled_feats = []
        sampled_hist_list = [0 for i in range(nBins)]
        sampled_hist = []
        curSamples = npoint
        curBins = nBins
        for i in range(nBins):
            npoint_bin = curSamples // curBins
            npoint_bin = min(npoint_bin, sort_hist[i][0].to(int))
            sampled_hist_list[i] = npoint_bin
            curSamples -= npoint_bin
            curBins -= 1
            sampled_hist.append(torch.Tensor([sort_hist[i][1], sort_hist[i][2], sampled_hist_list[i]]))
        sampled_hist = torch.stack(sampled_hist)
        hist, ind = sampled_hist.sort(0)
        sort_hist = sampled_hist[ind[...,0]]
        sampled_hist_list = sort_hist[...,2]
        
        # select points
        for i in range(shuffle_xyz.shape[0]):
            idx = min(torch.floor((shuffle_feat[i][0] - bin_min)/bin_width).to(int), nBins-1)
            if sampled_hist_list[idx] > 0:
                sampled_xyz.append(shuffle_xyz[i])
                sampled_feats.append(shuffle_feat[i])
                sampled_hist_list[idx] -= 1
            if torch.count_nonzero(sampled_hist_list) == 0:
                break
        sampled_xyz = torch.stack(sampled_xyz)
        sampled_feats = torch.stack(sampled_feats)
        sampled_xyz_batches.append(sampled_xyz)
        sampled_feats_batches.append(sampled_feats)
    sampled_xyz_batches = torch.stack(sampled_xyz_batches)
    sampled_feats_batches = torch.stack(sampled_feats_batches)
    return sampled_xyz_batches, sampled_feats_batches

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def clustered_idx(xyz, centroids, feat, centroid_feats, feat_weight):
    """
    Input:
        xyz: all points
        centroids: sampled points
        feat: all features
        centroid_feats: sampled points features
        feat_weight:
    Return:
        clustered_idx: clustered index, [B, N, 1]
    """
    sqrdists_spatial = square_distance(xyz, centroids)
    sqrdists_feature = square_distance(feat, centroid_feats)
    sqrdists = sqrdists_spatial + feat_weight * sqrdists_feature
    clustered_idx = torch.argmin(sqrdists, dim=2)
    return clustered_idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, sampling="FPS"):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if sampling == "FPS":
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
    else:
        new_xyz, feats = entropy_sampling(xyz, points, 16, npoint)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points

def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, sampling):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all
        self.sampling = sampling

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, False, self.sampling)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetEncoder(nn.Module):
    def __init__(self, in_channel, latent_size, sampling_method):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False, sampling=sampling_method)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False, sampling="FPS")
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True, sampling="FPS")
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_size)

    def forward(self, xyz, feat):
        assert xyz.shape[0] == feat.shape[0]
        assert xyz.shape[2] == feat.shape[2]
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
    

class PointNetPP_SR(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, pc_channel, latent_feats, sampling_method, pos_freq_const=10, pos_freq_num=30):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = PointNetEncoder(pc_channel, latent_feats, sampling_method)
        self.pos_features = in_features*pos_freq_num*2
        self.hidden_features = hidden_features
        self.latent_features = latent_feats
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + latent_feats, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, pc, pc_feats):
        pc_latent = self.lowres_encoding(pc, pc_feats)
        pos_latent = self.pos_encoding(coords)
        pc_latent = pc_latent.repeat(1, pos_latent.shape[1])
        pc_latent = pc_latent.reshape((pos_latent.shape[0], pos_latent.shape[1], self.latent_features))
        latent = torch.cat([pos_latent, pc_latent], dim=-1)
        output = self.net(latent)
        return output
    
    def get_latents(self, pc, pc_feats):
        pc_latent = self.lowres_encoding(pc, pc_feats)
        return pc_latent


class PointNetPP_SR_3D(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, out_features, pc_channel, latent_feats, sampling_method, pos_freq_const=10, pos_freq_num=30):
        super().__init__()
        self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
        self.lowres_encoding = PointNetEncoder(pc_channel, latent_feats, sampling_method)
        self.pos_features = in_features*pos_freq_num*3
        self.hidden_features = hidden_features
        self.latent_features = latent_feats
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        self.net = []
        self.net.append(MLPLayer(self.pos_features + latent_feats, hidden_features, self.relu))
        for i in range(hidden_layers-1):
            self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
        self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords, pc, pc_feats):
        pc_latent = self.lowres_encoding(pc, pc_feats)
        pos_latent = self.pos_encoding(coords)
        pc_latent = pc_latent.repeat(1, pos_latent.shape[1])
        pc_latent = pc_latent.reshape((pos_latent.shape[0], pos_latent.shape[1], self.latent_features))
        latent = torch.cat([pos_latent, pc_latent], dim=-1)
        output = self.net(latent)
        return output
    
    def get_latents(self, pc, pc_feats):
        pc_latent = self.lowres_encoding(pc, pc_feats)
        return pc_latent
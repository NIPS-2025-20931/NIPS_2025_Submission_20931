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

# class for fully nonequispaced 2d points
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat))
        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)
        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        return data_inv

class VFT3D:
    def __init__(self, x_positions, y_positions, z_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        x_positions -= torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        z_positions -= torch.min(z_positions)
        self.z_positions = z_positions * 6.28 / (torch.max(z_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()
        self.Z_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float().cuda()

        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2)*(self.modes*2)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2), 1)
        Y_mat = torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,(self.modes*2)*(self.modes*2),self.number_points)
        intermediate = (X_mat+Y_mat).repeat(1, self.modes*2, 1)
        Z_mat = (torch.bmm(self.Z_, self.z_positions[:,None,:]).repeat(1, 1, self.modes*2*self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (intermediate+Z_mat))
        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)
        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        return data_inv
    
class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, transformer):
        batchsize = x.shape[0]

        x = x.permute(0, 2, 1)
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes1-1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = x_ft[..., 2*self.modes1:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real

class SpectralConv3d_dse(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, dtype=torch.cfloat))

    def compl_mul3d(self, a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x, transformer):
        batchsize = x.shape[0]
        
        x = x.permute(0, 2, 1)
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes2, 2*self.modes3))

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels, x_ft.size(-3), x_ft.size(-2), x_ft.size(-1)//2, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = self.compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1 * 2*self.modes2 * self.modes3))
        x_ft2 = x_ft.flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real

class FNO_DSE_SR(nn.Module):
    def __init__(self, input_dim_edge, input_dim_node, num_modes, hidden_dim, output_dim_node):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=3, activation=nn.SiLU(), bias=True, activate_final=False)
        self.conv0 = SpectralConv2d_dse(hidden_dim, hidden_dim, num_modes, num_modes)
        self.conv1 = SpectralConv2d_dse(hidden_dim, hidden_dim, num_modes, num_modes)
        self.conv2 = SpectralConv2d_dse(hidden_dim, hidden_dim, num_modes, num_modes)
        self.conv3 = SpectralConv2d_dse(hidden_dim, hidden_dim, num_modes, num_modes)
        self.w0 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.decoder = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim_node, num_layers=3, activation=nn.SiLU(), bias=True, activate_final=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, transform):
        node_latents = self.encoder(node_features)

        node_latents = node_latents.permute(0, 2, 1)
        x1 = self.conv0(node_latents, transform)
        x2 = self.w0(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv1(node_latents, transform)
        x2 = self.w1(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv2(node_latents, transform)
        x2 = self.w2(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv3(node_latents, transform)
        x2 = self.w3(node_latents)
        node_latents = x1 + x2
        node_latents = node_latents.permute(0, 2, 1)

        node_features = self.sigmoid(self.decoder(node_latents))

        return node_features


class FNO_DSE_SR_3D(nn.Module):
    def __init__(self, input_dim_edge, input_dim_node, num_modes, hidden_dim, output_dim_node):
        super().__init__()
        self.encoder = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=3, activation=nn.SiLU(), bias=True, activate_final=False)
        self.conv0 = SpectralConv3d_dse(hidden_dim, hidden_dim, num_modes, num_modes, num_modes)
        self.conv1 = SpectralConv3d_dse(hidden_dim, hidden_dim, num_modes, num_modes, num_modes)
        self.conv2 = SpectralConv3d_dse(hidden_dim, hidden_dim, num_modes, num_modes, num_modes)
        self.conv3 = SpectralConv3d_dse(hidden_dim, hidden_dim, num_modes, num_modes, num_modes)
        self.w0 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.w3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.decoder = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim_node, num_layers=3, activation=nn.SiLU(), bias=True, activate_final=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, transform):
        node_latents = self.encoder(node_features)

        node_latents = node_latents.permute(0, 2, 1)
        x1 = self.conv0(node_latents, transform)
        x2 = self.w0(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv1(node_latents, transform)
        x2 = self.w1(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv2(node_latents, transform)
        x2 = self.w2(node_latents)
        node_latents = x1 + x2
        node_latents = F.gelu(node_latents)

        x1 = self.conv3(node_latents, transform)
        x2 = self.w3(node_latents)
        node_latents = x1 + x2
        node_latents = node_latents.permute(0, 2, 1)

        node_features = self.sigmoid(self.decoder(node_latents))

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

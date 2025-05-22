import numpy as np
import torch
import torch.nn as nn
import json
from PIL import Image
from scipy.spatial import cKDTree

xyres = {'backwardStep': (24, 120), 'cylinderArray': (16, 16)}
dataidxrange = {'backwardStep': {'train': (0, 64), 'test': (0, 17)} , 'cylinderArray': {'train': (0, 96), 'test': (96, 128)}, 'shallow_water': {'train': (0, 32), 'test': (0, 8)}, 'smoke': {'train': (0, 32), 'test':(0, 9)}}

lprange = {'backwardStep': (-92.57, 20.80), 'cylinderArray': (-1.3047, 0.6740), 'shallow_water': (-0.3065, 0.3116), 'smoke': (0.0, 6.0474)}
hprange = {'backwardStep': (-113.44, 50.78), 'cylinderArray': (-2.3735, 1.3202), 'shallow_water': (-0.3065, 0.3116), 'smoke': (0.0, 6.0474)}

def pytorch_device_config(device_num=0):
    #  configuring device
    if torch.cuda.is_available():
        device = torch.device('cuda:'+str(device_num))
        print('Running on the GPU')
    else:
        device = torch.device('cpu')
        print('Running on the CPU')
    return device

def load_config(filename):
    """Load a configuration file."""
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def save_config(filename, config):
    """Save a configuration file. """
    with open(filename, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

def turbulent_load(fname):
    pressure = LoadPressure(fname)
    pressure = pressure.reshape(pressure.shape[0], pressure.shape[1]*pressure.shape[2])
    pressure = torch.from_numpy(pressure)
    return pressure

@torch.jit.script
def positional_encoding(
        v: torch.Tensor,
        sigma: float,
        m: int) -> torch.Tensor:
    r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`
        where :math:`j \in \{0, \dots, m-1\}`

    Args:
        v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`
        sigma (float): constant chosen based upon the domain of :attr:`v`
        m (int): [description]

    Returns:
        Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`

    See :class:`~rff.layers.PositionalEncoding` for more details.
    """
    j = torch.arange(m, device=v.device)
    coeffs = 2 * np.pi * sigma ** (j / m)
    vp = coeffs * torch.unsqueeze(v, -1)
    vp_cat = torch.cat((torch.cos(vp), torch.sin(vp)), dim=-1)
    return vp_cat.flatten(-2, -1)

class PositionalEncoding(nn.Module):
    """Layer for mapping coordinates using the positional encoding"""

    def __init__(self, sigma: float, m: int):
        r"""
        Args:
            sigma (float): frequency constant
            m (int): number of frequencies to map to
        """
        super().__init__()
        self.sigma = sigma
        self.m = m

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        r"""Computes :math:`\gamma(\mathbf{v}) = (\dots, \cos{2 \pi \sigma^{(j/m)} \mathbf{v}} , \sin{2 \pi \sigma^{(j/m)} \mathbf{v}}, \dots)`

        Args:
            v (Tensor): input tensor of shape :math:`(N, *, \text{input_size})`

        Returns:
            Tensor: mapped tensor of shape :math:`(N, *, 2 \cdot m \cdot \text{input_size})`
        """
        return positional_encoding(v, self.sigma, self.m)


# class SineLayer(nn.Module):
#     # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
#     # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
#     # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
#     # hyperparameter.
    
#     # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
#     # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
#     def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = omega_0
#         self.is_first = is_first
        
#         self.in_features = in_features
#         self.linear = nn.Linear(in_features, out_features, bias=bias)
        
#         self.init_weights()
    
#     def init_weights(self):
#         with torch.no_grad():
#             if self.is_first:
#                 self.linear.weight.uniform_(-1 / self.in_features, 
#                                              1 / self.in_features)      
#             else:
#                 self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
#                                              np.sqrt(6 / self.in_features) / self.omega_0)
        
#     def forward(self, input):
#         return torch.sin(self.omega_0 * self.linear(input))
    
# class ResidualSineLayer(nn.Module):
#     def __init__(self, features, bias=True, ave_first=False, ave_second=False, omega_0=30):
#         super().__init__()
#         self.omega_0 = omega_0

#         self.features = features
#         self.linear_1 = nn.Linear(features, features, bias=bias)
#         self.linear_2 = nn.Linear(features, features, bias=bias)

#         self.weight_1 = .5 if ave_first else 1
#         self.weight_2 = .5 if ave_second else 1

#         self.init_weights()

#     def init_weights(self):
#         with torch.no_grad():
#             self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
#                                            np.sqrt(6 / self.features) / self.omega_0)
#             self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0, 
#                                            np.sqrt(6 / self.features) / self.omega_0)
#     def forward(self, input):
#         sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
#         sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
#         return self.weight_2*(input+sine_2)


class MLPLayer(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # torch.nn.init.xavier_normal_(self.linear.weight)
        # if bias:
        #     torch.nn.init.normal_(self.linear.bias, 0, 0.001)
        self.act = activation_fn
    
    def forward(self, x):
        if self.act is None:
            return self.linear(x)
        return self.act(self.linear(x))



# class ConcatINR(nn.Module):
#     def __init__(self, in_features, out_features, latent_features, hidden_features, hidden_layers, pos_freq_const=10, pos_freq_num=30):
#         super().__init__()
#         self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
#         self.pos_features = in_features*pos_freq_num*2
#         self.latent_features = latent_features
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#         self.net = []
#         self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
#         for i in range(hidden_layers-1):
#             self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
#         self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
#         self.net = nn.Sequential(*self.net)

#     def forward(self, latents, coords):
#         # latents: [Batch, latent_dim]
#         # coords: [Batch, points, 3]
#         pos_latent = self.pos_encoding(coords)
#         latents = latents.repeat(1, pos_latent.shape[1])
#         latents = latents.reshape((coords.shape[0], pos_latent.shape[1], self.latent_features))
#         latents = torch.cat([pos_latent, latents], dim=-1)
#         output = self.net(latents)
#         return output

# class TemporalAttnINR(nn.Module):
#     def __init__(self, in_features, out_features, num_heads, ntimesteps, latent_features, hidden_features, hidden_layers, pos_freq_const=10, pos_freq_num=30):
#         super().__init__()
#         self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
#         self.pos_features = in_features*pos_freq_num*2
#         self.latent_features = latent_features
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#         self.w_qs = nn.Linear(latent_features, latent_features, bias=False)
#         self.w_ks = nn.Linear(latent_features, latent_features, bias=False)
#         self.w_vs = nn.Linear(latent_features, latent_features, bias=False)

#         self.multihead_attn = nn.MultiheadAttention(latent_features, num_heads, batch_first=True)
#         self.fc = MLPLayer(ntimesteps*latent_features, latent_features, self.relu)
#         self.net = []
#         self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
#         for i in range(hidden_layers-1):
#             self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
#         self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
#         self.net = nn.Sequential(*self.net)


#     def forward(self, latents, coords):
#         # latents: [Batch, timesteps, latent_dim]
#         # coords: [Batch, points, 3]
#         Q, K, V = self.w_qs(latents), self.w_ks(latents), self.w_vs(latents)
#         attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
#         concat_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1]*self.latent_features)
#         pred_latent = self.fc(concat_output)
#         pos_latent = self.pos_encoding(coords)
#         pred_latent = pred_latent.repeat(1, pos_latent.shape[1])
#         pred_latent = pred_latent.reshape((coords.shape[0], pos_latent.shape[1], self.latent_features))
#         latents = torch.cat([pos_latent, pred_latent], dim=-1)
#         output = self.net(latents)
#         return output
    

# class TemporalAttnINRv2(nn.Module):
#     def __init__(self, in_features, out_features, num_heads, ntimesteps, latent_features, hidden_features, hidden_layers, pos_freq_const=10, pos_freq_num=30):
#         super().__init__()
#         self.pos_encoding = PositionalEncoding(pos_freq_const, pos_freq_num)
#         self.pos_features = in_features*pos_freq_num*2
#         self.latent_features = latent_features
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#         self.w_qs = nn.Linear(latent_features, latent_features, bias=False)
#         self.w_ks = nn.Linear(latent_features, latent_features, bias=False)
#         self.w_vs = nn.Linear(latent_features, latent_features, bias=False)

#         self.multihead_attn = nn.MultiheadAttention(latent_features, num_heads, batch_first=True)
#         self.fc1 = MLPLayer(ntimesteps*latent_features, ntimesteps*latent_features, self.relu)
#         self.fc2 = MLPLayer(ntimesteps*latent_features, latent_features, self.relu)
#         self.net = []
#         self.net.append(MLPLayer(self.pos_features + latent_features, hidden_features, self.relu))
#         for i in range(hidden_layers-1):
#             self.net.append(MLPLayer(hidden_features, hidden_features, self.relu))
#         self.net.append(MLPLayer(hidden_features, out_features, self.sigmoid))
#         self.net = nn.Sequential(*self.net)


#     def forward(self, latents, coords):
#         # latents: [Batch, timesteps, latent_dim]
#         # coords: [Batch, points, 3]
#         Q, K, V = self.w_qs(latents), self.w_ks(latents), self.w_vs(latents)
#         attn_output, attn_output_weights = self.multihead_attn(Q, K, V)
#         concat_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1]*self.latent_features)
#         pred_latent = self.fc2(self.fc1(concat_output))
#         pos_latent = self.pos_encoding(coords)
#         pred_latent = pred_latent.repeat(1, pos_latent.shape[1])
#         pred_latent = pred_latent.reshape((coords.shape[0], pos_latent.shape[1], self.latent_features))
#         latents = torch.cat([pos_latent, pred_latent], dim=-1)
#         output = self.net(latents)
#         return output



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


def sq_dist(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [N, C]
        dst: target points, [M, C]
    Output:
        dist: per-point square distance, [N, M]
    """
    return torch.sum((src[:, None, :] - dst[None, :, :]) ** 2, dim=-1)

def turbulent_flow_sampling(dataset, sampling_rate: float, is_graph=False):
    Ny, Nx = xyres[dataset][0], xyres[dataset][1]
    norm = max(Nx, Ny) - 1
    lowres_pos = np.zeros((Nx*Ny, 2), dtype=np.float32)
    for i in range(Ny):
        for j in range(Nx):
            lowres_pos[i*Nx+j][0] = (float)(i/norm)
            lowres_pos[i*Nx+j][1] = (float)(j/norm)

    lowres_pos_t = torch.from_numpy(lowres_pos)

    if sampling_rate > 1.0 or sampling_rate <= 0.0:
        raise ValueError("Sampling Rate should be in range (0, 1]")
    elif sampling_rate < 1.0:
        nSamples = int(Nx*Ny*sampling_rate)
        shuffle_arr = torch.randperm(Nx*Ny)
        selected_samples = shuffle_arr[:nSamples]
        lowres_pos_t = lowres_pos_t[selected_samples]
    else:
        nSamples = Nx*Ny
        selected_samples = torch.arange(0, Nx*Ny, dtype=torch.int64)

    if is_graph:
        N = nSamples
        # Not sure how to set radius value
        radius = (1.0/norm) * (0.01 + 1.0/(sampling_rate**2))
        sd = sq_dist(lowres_pos_t, lowres_pos_t)
        group_idx = torch.arange(N, dtype=torch.long).view(1, N).repeat([N, 1])
        group_idx[sd > radius ** 2] = N
        group_idx = group_idx.sort(dim=-1)[0]
        edge_index_list = []
        for i in range(len(group_idx)):
            for j in range(len(group_idx)):
                if group_idx[i][j] == i:
                    continue
                if group_idx[i][j] == N:
                    break
                edge_index_list.append([i, group_idx[i][j].item()])
        edge_index_t = torch.tensor(edge_index_list, dtype=torch.long)
        edge_index = edge_index_t.t()
        return lowres_pos_t, selected_samples, edge_index
    
    return lowres_pos_t, selected_samples


# def bilinear_interpolation_gpu(grid, values, x_range, y_range, points, device="cuda"):
#     """
#     Perform bilinear interpolation for given points in a 2D scalar field using PyTorch on GPU.

#     Parameters:
#     - grid: tuple (M, N), size of the regular grid.
#     - values: torch.Tensor of shape (M, N), scalar values at grid points.
#     - x_range: tuple (a, b), range of x-coordinates.
#     - y_range: tuple (c, d), range of y-coordinates.
#     - points: torch.Tensor of shape (K, 2), query points.

#     Returns:
#     - interpolated_values: torch.Tensor of shape (K,), interpolated values at query points.
#     """
#     M, N = grid
#     a, b = x_range
#     c, d = y_range

#     # Move to GPU
#     values = values.to(device)
#     points = points.to(device)

#     # Compute grid spacing
#     dx = (b - a) / (M - 1)
#     dy = (d - c) / (N - 1)

#     # Compute indices of the four surrounding grid points
#     x_idx = (points[:, 0] - a) / dx
#     y_idx = (points[:, 1] - c) / dy

#     x0 = torch.floor(x_idx).long().clamp(0, M - 2)
#     y0 = torch.floor(y_idx).long().clamp(0, N - 2)
#     x1 = (x0 + 1).clamp(0, M - 1)
#     y1 = (y0 + 1).clamp(0, N - 1)

#     # Get grid point values
#     Q00 = values[x0, y0]
#     Q01 = values[x0, y1]
#     Q10 = values[x1, y0]
#     Q11 = values[x1, y1]

#     # Compute interpolation weights
#     x_frac = x_idx - x0.float()
#     y_frac = y_idx - y0.float()

#     # Bilinear interpolation
#     interpolated_values = (
#         Q00 * (1 - x_frac) * (1 - y_frac) +
#         Q10 * x_frac * (1 - y_frac) +
#         Q01 * (1 - x_frac) * y_frac +
#         Q11 * x_frac * y_frac
#     )

#     return interpolated_values


def batch_bilinear_interpolation_gpu(grid, values, x_range, y_range, points, device="cuda"):
    """
    Perform bilinear interpolation for given points in a 2D scalar field using PyTorch on GPU.

    Parameters:
    - grid: tuple (M, N), size of the regular grid.
    - values: torch.Tensor of shape (M, N), scalar values at grid points.
    - x_range: tuple (a, b), range of x-coordinates.
    - y_range: tuple (c, d), range of y-coordinates.
    - points: torch.Tensor of shape (K, 2), query points.

    Returns:
    - interpolated_values: torch.Tensor of shape (K,), interpolated values at query points.
    """
    M, N = grid
    a, b = x_range
    c, d = y_range

    # Move to GPU
    values = values.to(device)
    points = points.to(device)

    # Compute grid spacing
    dx = (b - a) / (M - 1)
    dy = (d - c) / (N - 1)

    # Compute indices of the four surrounding grid points
    x_idx = (points[:, 0] - a) / dx
    y_idx = (points[:, 1] - c) / dy

    x0 = torch.floor(x_idx).long().clamp(0, M - 2)
    y0 = torch.floor(y_idx).long().clamp(0, N - 2)
    x1 = (x0 + 1).clamp(0, M - 1)
    y1 = (y0 + 1).clamp(0, N - 1)

    # Get grid point values
    Q00 = values[:, x0, y0]
    Q01 = values[:, x0, y1]
    Q10 = values[:, x1, y0]
    Q11 = values[:, x1, y1]

    # Compute interpolation weights
    x_frac = x_idx - x0.float()
    y_frac = y_idx - y0.float()

    # Bilinear interpolation
    interpolated_values = (
        Q00 * (1 - x_frac) * (1 - y_frac) +
        Q10 * x_frac * (1 - y_frac) +
        Q01 * (1 - x_frac) * y_frac +
        Q11 * x_frac * y_frac
    )

    return interpolated_values


# def cuda_memory_usage():
#     freemem = torch.cuda.mem_get_info()[0]
#     print('Free memory: {0:.4f} GB'.format(freemem/(1024*1024*1024)))

def build_xyz_vertices(phi, theta):
    phi = phi.ravel()
    theta = theta.ravel()
    expanded_phi, expanded_theta = np.meshgrid(phi, theta, indexing='ij')
    x = np.sin(expanded_theta) * np.cos(expanded_phi)
    y = np.sin(expanded_theta) * np.sin(expanded_phi)
    z = np.cos(expanded_theta)
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)
    x, y, z = (x-xmin) / (xmax-xmin), (y-ymin) / (ymax-ymin), (z-zmin) / (zmax-zmin)
    xyz_coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    return expanded_phi, expanded_theta, xyz_coords

def max_closest_dist(points):
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest_distances = distances[:, 1]
    return np.max(nearest_distances)

def All2AllPointCloudToGraph(points, radius):
    tree = cKDTree(points)
    pairs = tree.query_ball_tree(tree, radius)

    edges = []
    for i, neighbors in enumerate(pairs):
        for j in neighbors:
            if i < j:
                edges.append([i, j])

    edges = np.array(edges)
    bidirectional_edges = np.vstack([edges, np.flip(edges, axis=1)])

    return bidirectional_edges

def to_scientific(x):
    if x == 0:
        return 0.0, 0
    b = np.floor(np.log10(abs(x)))
    a = x / (10 ** b)
    return a, b


def build_s2_coord_vertices(phi, theta):
    phi = phi.ravel()
    phi_vert = np.concatenate([phi, [2*np.pi]])
    phi_vert -= phi_vert[1] / 2
    theta = theta.ravel()
    theta_mid = (theta[:-1] + theta[1:]) / 2
    theta_vert = np.concatenate([[np.pi], theta_mid, [0]])
    return np.meshgrid(phi_vert, theta_vert, indexing='ij')

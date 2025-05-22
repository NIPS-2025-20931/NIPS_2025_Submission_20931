import sys
sys.path.append("../model")
from FLUID import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

def generate_grids(width, height, num_samples):
    # Approximate grids
    total_area = width * height
    grid_size = torch.sqrt(torch.tensor(total_area / num_samples))

    # Generate grid points
    x_coords = torch.arange(0, width, grid_size)
    y_coords = torch.arange(0, height, grid_size)
    xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
    grid_points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    return grid_size, grid_points

def jittered_grid_sampling_fixed(grid_size, grid_points, num_samples):
    # jitter grid points
    jitter = torch.rand(grid_points.shape) * grid_size
    points = grid_points + jitter

    # randomly delete some points
    if len(points) > num_samples:
        selected_indices = torch.randperm(len(points))[:num_samples]
        points = points[selected_indices]
    return points

def samplepoints2graphs(args, device, points, radius):
    index_arr = np.arange(len(points))
    points, reordered_index, npts = MultiResGraph(points=points.numpy(), index=index_arr, levels=3, pc2gfactor=3.0, g2gfactor=4.0, method='random')
    nodeidx = np.arange(npts[0])
    pc2g_edges = PointCloudToGraph(points, nodeidx, radius=radius)
    g2g_edges = GraphToGraphEdges(points, npts)
    g2pc_edges = np.flip(pc2g_edges, axis=-1)
    g2pc_edges = g2pc_edges.copy()
    
    points = torch.from_numpy(points)
    pc2g_edges = torch.from_numpy(pc2g_edges)
    g2g_edges = torch.from_numpy(g2g_edges)
    g2pc_edges = torch.from_numpy(g2pc_edges)

    # Encoder
    senders = pc2g_edges[...,0]
    receivers = pc2g_edges[...,1]
    relative_mesh_pos = points[senders] - points[receivers]
    pc2g_edge_features = torch.cat([relative_mesh_pos / radius, torch.norm(relative_mesh_pos, dim=-1, keepdim=True) / radius], dim=-1)

    # DSE
    g_points = points[:npts[0]][None,...].to(device)
    g_points = g_points.repeat(args.batch_size, 1, 1)
    transform = VFT(g_points[:,:,0], g_points[:,:,1], args.num_modes)

    # Multi-Res Graph
    senders = g2g_edges[...,0]
    receivers = g2g_edges[...,1]
    relative_mesh_pos = points[senders] - points[receivers]
    g2g_edge_features = torch.cat([relative_mesh_pos, torch.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)

    # Decoder
    senders = g2pc_edges[...,0]
    receivers = g2pc_edges[...,1]
    relative_mesh_pos = points[senders] - points[receivers]
    g2pc_edge_features = torch.cat([relative_mesh_pos / radius, torch.norm(relative_mesh_pos, dim=-1, keepdim=True) / radius], dim=-1)

    # Data Preparation
    pc2g_edges = pc2g_edges.to(device)
    g2g_edges = g2g_edges.to(device)
    g2pc_edges = g2pc_edges.to(device)

    pc2g_edge_features = pc2g_edge_features[None,...]
    pc2g_edge_features = pc2g_edge_features.repeat(args.batch_size, 1, 1)
    pc2g_edge_features = pc2g_edge_features.to(device)

    g2g_edge_features = g2g_edge_features[None,...]
    g2g_edge_features = g2g_edge_features.repeat(args.batch_size, 1, 1)
    g2g_edge_features = g2g_edge_features.to(device)

    g2pc_edge_features = g2pc_edge_features[None,...]
    g2pc_edge_features = g2pc_edge_features.repeat(args.batch_size, 1, 1)
    g2pc_edge_features = g2pc_edge_features.to(device)

    pc2g = {}
    g2g = {}
    g2pc = {}
    pc2g['edge_idx'] = pc2g_edges
    pc2g['edge_features'] = pc2g_edge_features

    g2g['edge_idx'] = g2g_edges
    g2g['edge_features'] = g2g_edge_features
    g2g['node_features'] = None

    g2pc['edge_idx'] = g2pc_edges
    g2pc['edge_features'] = g2pc_edge_features
    g2pc['node_features'] = None

    return points, reordered_index, pc2g, g2g, transform, g2pc


@hydra.main(version_base=None, config_path="conf", config_name="FLUID_train")
def main(args : DictConfig):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pytorch_device_config()

    data_indices = np.arange(0, dataidxrange[args.dataset]['train'][1]*80, dtype=int)

    input_dim_node=1
    input_dim_edge=3
    out_features = 1

    pos_t, selected_samples = turbulent_flow_sampling(args.dataset, args.input_sampling_rate, False)
    pos = pos_t.numpy()
    num_samples = len(selected_samples)

    LRx, LRy = xyres[args.dataset][0], xyres[args.dataset][1]
    LRnorm = max(LRx, LRy) - 1
    LRxmax, LRymax = (float)(LRx/(LRnorm+1)), (float)(LRy/(LRnorm+1))
    total_area = LRxmax * LRymax
    radius = torch.sqrt(torch.tensor(total_area / num_samples)) * 3.0

    grid_sample_points, reordered_index, grid_pc2g, grid_g2g, grid_transform, grid_g2pc = samplepoints2graphs(args, device, pos_t, radius)

    ####################################################################################################
    model = FLUID(input_dim_edge=input_dim_edge, input_dim_node=input_dim_node, num_modes=args.num_modes, hidden_dim=args.latent_feats, output_dim_node=out_features)
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(args.dir_weights, args.network_str + '_'+ str(args.start_epoch) + ".pth"), weights_only=True))
    model.to(device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    size_all_kb = (param_size) / 1024
    print('Model size: {:.3f} KB'.format(size_all_kb))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.loss == 'MSE':
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()
    elif args.loss == 'L1':
        print('Use L1 Loss')
        criterion = torch.nn.L1Loss()
    else:
        print('Use MSE Loss')
        criterion = torch.nn.MSELoss()

    ####################################################################################################
    # Load all data
    all_lf_pressure = None
    all_hf_pressure = None
    for i in range(*dataidxrange[args.dataset]['train']):
        lf_fname = args.root + 'low_fidelity_' + str(i) + '.npy'
        hf_fname = args.root + 'high_fidelity_' + str(i) + '.npy'
        lf_p = torch.from_numpy(np.load(lf_fname)).reshape(-1, xyres[args.dataset][0]*xyres[args.dataset][1])
        hf_p = torch.from_numpy(np.load(hf_fname)).reshape(-1, xyres[args.dataset][0]*xyres[args.dataset][1])
        if all_lf_pressure is None:
            all_lf_pressure = lf_p
            all_hf_pressure = hf_p
        else:
            all_lf_pressure = torch.cat([all_lf_pressure, lf_p], dim=0)
            all_hf_pressure = torch.cat([all_hf_pressure, hf_p], dim=0)
    all_lf_pressure = ((all_lf_pressure - lprange[args.dataset][0]) / (lprange[args.dataset][1] - lprange[args.dataset][0]))
    all_hf_pressure = ((all_hf_pressure - hprange[args.dataset][0]) / (hprange[args.dataset][1] - hprange[args.dataset][0]))

    all_lf_pressure = all_lf_pressure[:, selected_samples][..., reordered_index][...,None].to(device)
    all_hf_pressure = all_hf_pressure[:, selected_samples][..., reordered_index][...,None].to(device)

    ####################################################################################################

    losses = []
    total_training_time_start = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        total_loss = 0
        errsum = 0
        tstart = time.time()
        np.random.shuffle(data_indices)
        for tidx in range(math.ceil(len(data_indices)/args.batch_size)):
            ts = data_indices[tidx*args.batch_size:(tidx+1)*args.batch_size]
            lf_feats = all_lf_pressure[ts]
            hf_feats = all_hf_pressure[ts]
            grid_pc2g['node_features'] = lf_feats
            # ===================forward=====================
            model_output = model(grid_pc2g, grid_g2g, grid_transform, grid_g2pc)
            loss = criterion(model_output, hf_feats)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_mean_loss = loss.data.cpu().numpy()
            errsum += batch_mean_loss * num_samples * args.batch_size
            total_loss += batch_mean_loss * args.batch_size
        tend = time.time()
        mse = errsum / (num_samples * len(data_indices))
        curr_psnr = 20. * np.log10(1.0) - 10. * np.log10(mse)
        losses.append(total_loss/len(data_indices))
        print('Training time: {0:.4f} for {1} data points x {2} timesteps, approx PSNR = {3:.4f}, approx MSE = {4:.8f} * 10^{5}'\
                  .format(tend-tstart, num_samples, len(data_indices), curr_psnr, *to_scientific(np.mean(losses))))
        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(args.dir_weights, args.network_str + '_'+ str(epoch+1) + ".pth"))
    
    print('Total training time: {0:.4f}'.format(tend-total_training_time_start))

if __name__ == '__main__':
    main()

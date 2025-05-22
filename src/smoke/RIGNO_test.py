import sys
sys.path.append("../model")
from RIGNO import *
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

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
    pc2g_edge_features = pc2g_edge_features.to(device)

    g2g_edge_features = g2g_edge_features[None,...]
    g2g_edge_features = g2g_edge_features.to(device)

    g2pc_edge_features = g2pc_edge_features[None,...]
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

    return points, reordered_index, pc2g, g2g, g2pc



@hydra.main(version_base=None, config_path="conf", config_name="RIGNO_test")
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

    coords = np.load(args.root + 'xy.npy')

    # The domain is defined on [0, 100] x [0, 100]
    coords = (coords / 100)
    radius = max_closest_dist(coords) * 3.0

    pos_t = torch.from_numpy(coords.astype(np.float32))
    num_nodes = pos_t.shape[0]

    grid_sample_points, reordered_index, grid_pc2g, grid_g2g, grid_g2pc = samplepoints2graphs(args, device, pos_t, radius)

    ####################################################################################################

    model = RIGNO_SR(input_dim_edge=input_dim_edge, input_dim_node=input_dim_node, hidden_dim=args.latent_feats, output_dim_node=out_features)
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(args.dir_weights, args.network_str + '_'+ str(args.start_epoch) + ".pth"), weights_only=True))
    model.to(device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    size_all_kb = (param_size) / 1024
    print('Model size: {:.3f} KB'.format(size_all_kb))

    ####################################################################################################
    all_lf_smoke = None
    all_hf_smoke = None
    for i in range(*dataidxrange[args.dataset]['test']):
        lf_fname = args.root + 'low_fidelity_{0}.npy'.format(i)
        hf_fname = args.root + 'high_fidelity_{0}.npy'.format(i)
        lf_s = torch.from_numpy(np.load(lf_fname)[81:].astype(np.float32))
        hf_s = torch.from_numpy(np.load(hf_fname)[81:].astype(np.float32))
        lf_s = ((lf_s - lprange[args.dataset][0]) / (lprange[args.dataset][1] - lprange[args.dataset][0]))
        hf_s = ((hf_s - hprange[args.dataset][0]) / (hprange[args.dataset][1] - hprange[args.dataset][0]))
        if all_lf_smoke is None:
            all_lf_smoke = lf_s
            all_hf_smoke = hf_s
        else:
            all_lf_smoke = torch.cat([all_lf_smoke, lf_s], dim=0)
            all_hf_smoke = torch.cat([all_hf_smoke, hf_s], dim=0)

    all_lf_smoke = all_lf_smoke[..., reordered_index][...,None].to(device)
    all_hf_smoke = all_hf_smoke[..., reordered_index][...,None].to(device)

    ####################################################################################################
    #test
    mask = np.load(args.root + 'mask.npy')
    mask_flatten = mask.flatten()
    selected_samples = np.arange(len(mask_flatten))[mask_flatten]

    model.eval()
    psnrs = []
    msearr = []
    maearr = []
    with torch.no_grad():
        errsum = 0
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            curpsnr = 0.0
            for j in range(80):
                lf_feats = all_lf_smoke[i*80+j][None,...]
                hf_feats = all_hf_smoke[i*80+j][None,...]
                grid_pc2g['node_features'] = lf_feats
                # ===================forward=====================
                model_output = model(grid_pc2g, grid_g2g, grid_g2pc)
                model_output, hf_feats = torch.squeeze(model_output), torch.squeeze(hf_feats)
                mse = torch.sum((model_output-hf_feats)**2) / len(model_output)
                mae = torch.sum(torch.abs(model_output-hf_feats)) / len(model_output)
                mse = mse.data.cpu().numpy()
                mae = mae.data.cpu().numpy()
                msearr.append(mse)
                maearr.append(mae)
                errsum += mse
                psnr = 20. * np.log10(1.0) - 10. * np.log10(mse)
                curpsnr += psnr

                model_output = model_output.cpu().numpy()
                fullres_arr = np.full(mask.shape[0]*mask.shape[1], np.nan)
                fullres_arr[selected_samples[reordered_index]] = model_output
                plt.imshow(fullres_arr.reshape(mask.shape[0], mask.shape[1]).T, origin='lower', cmap='PRGn', vmin=0.0, vmax=1.0)
                plt.tight_layout()
                plt.axis('off')
                plt.savefig(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j+1), dpi=300, bbox_inches='tight', pad_inches=0)
                plt.clf()

            psnrs.append(curpsnr/80)
            with imageio.get_writer(args.dir_outputs + args.network_str + "_{0}.gif".format(i), mode='I') as writer:
                for j in range(80):
                    image = imageio.imread(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j+1))
                    writer.append_data(image)
            for j in range(80):
                if (j+1) % 10 != 0:
                    os.remove(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j+1))
        errsum /= ((dataidxrange[args.dataset]['test'][1] - dataidxrange[args.dataset]['test'][0])*80)
        psnr = 20. * np.log10(1.0) - 10. * np.log10(errsum)
        print('=======> Average PSNR = {0:.4f}, MSE = {1:.8f} * 10^{2}, MAE = {3:.8f} * 10^{4}'.format(psnr, *to_scientific(np.mean(msearr)), *to_scientific(np.mean(maearr))))
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            print('Test File Index: {0}, PSNR = {1:.4f}'.format(i, psnrs[i]))

if __name__ == '__main__':
    main()

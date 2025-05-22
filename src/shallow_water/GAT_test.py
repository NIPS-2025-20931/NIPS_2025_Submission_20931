import sys
sys.path.append("../model")
from GAT import *
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import imageio.v2 as imageio
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="GAT_test")
def main(args : DictConfig):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pytorch_device_config()

    in_features = 2
    out_features = 1
    pos_freq_num = 30

    phi = np.load(args.root + 'phi.npy')
    theta = np.load(args.root + 'theta.npy')

    expanded_phi, expanded_theta, xyz_coords = build_xyz_vertices(phi, theta)
    radius = max_closest_dist(xyz_coords) * 2.0
    edge_index = All2AllPointCloudToGraph(xyz_coords, radius)

    batch_pos = torch.from_numpy(xyz_coords.astype(np.float32))
    batch_pos = batch_pos[None,...]
    batch_pos = batch_pos.to(device)

    edge_index = torch.from_numpy(edge_index.astype(np.int16)).t()
    edge_index = edge_index[None,...]
    edge_index = edge_index.to(device)

    num_samples = batch_pos.shape[1]

    phi_vert, theta_vert = build_s2_coord_vertices(phi, theta)
    npx = np.sin(theta_vert) * np.cos(phi_vert)
    npy = np.sin(theta_vert) * np.sin(phi_vert)
    npz = np.cos(theta_vert)

    ####################################################################################################

    model = GAT_SR_3D(in_features=in_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features, \
                        node_dim=1, latent_features=args.latent_feats, msg_passing_steps=args.msg_passing_steps, \
                        graph_nodes=num_samples, pos_freq_const=10, pos_freq_num=pos_freq_num, pooling=args.latent_fuse)
    model.load_state_dict(torch.load(os.path.join(args.dir_weights, args.network_str + '_'+ str(args.start_epoch) + ".pth"), weights_only=True))
    model.to(device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    size_all_kb = (param_size) / 1024
    print('Model size: {:.3f} KB'.format(size_all_kb))

    ####################################################################################################
    # Load all data
    all_lf_height = []
    all_hf_height = []
    for i in range(*dataidxrange[args.dataset]['test']):
        lf_fname = args.root + 'low_fidelity_' + str(i) + '.npy'
        hf_fname = args.root + 'high_fidelity_' + str(i) + '.npy'
        lf_h = np.load(lf_fname)
        hf_h = np.load(hf_fname)
        all_lf_height.append(lf_h)
        all_hf_height.append(hf_h)
    all_lf_height = np.stack(all_lf_height, axis=0)
    all_hf_height = np.stack(all_hf_height, axis=0)
    B = all_lf_height.shape[0] * all_lf_height.shape[1]
    N = all_hf_height.shape[2] * all_hf_height.shape[3]
    all_lf_height = all_lf_height.reshape(B, N, 1)
    all_hf_height = all_hf_height.reshape(B, N, 1)
    all_lf_height = ((all_lf_height - lprange[args.dataset][0]) / (lprange[args.dataset][1] - lprange[args.dataset][0]))
    all_hf_height = ((all_hf_height - hprange[args.dataset][0]) / (hprange[args.dataset][1] - hprange[args.dataset][0]))
    all_lf_height = torch.from_numpy(all_lf_height.astype(np.float32)).to(device)
    all_hf_height = torch.from_numpy(all_hf_height.astype(np.float32)).to(device)

    ####################################################################################################

    #test
    cmap = plt.cm.seismic
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    norm = matplotlib.colors.Normalize(0, 1)

    psnrs = []
    msearr = []
    maearr = []
    model.eval()
    with torch.no_grad():
        errsum = 0
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            curpsnr = 0.0
            for j in range(12):
                lf_feats = all_lf_height[i*12+j][None,...]
                hf_feats = all_hf_height[i*12+j][None,...]
                # ===================forward=====================
                model_output = model(batch_pos, edge_index, lf_feats)
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
                fc = cmap(norm(model_output.reshape(len(phi), len(theta[0]))))
                surf = ax.plot_surface(npx, npy, npz, facecolors=fc, cstride=1, rstride=1, linewidth=0, antialiased=False, shade=False, zorder=5)
                ax.set_box_aspect((1,1,1))
                ax.set_xlim(-0.7, 0.7)
                ax.set_ylim(-0.7, 0.7)
                ax.set_zlim(-0.7, 0.7)
                ax.axis('off')
                fig.savefig(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j), dpi=100)

            psnrs.append(curpsnr/12)
            with imageio.get_writer(args.dir_outputs + args.network_str + "_{0}.gif".format(i), mode='I') as writer:
                for j in range(12):
                    image = imageio.imread(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j))
                    writer.append_data(image)
            for j in range(12):
                os.remove(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j))
        errsum /= ((dataidxrange[args.dataset]['test'][1] - dataidxrange[args.dataset]['test'][0])*12)
        psnr = 20. * np.log10(1.0) - 10. * np.log10(errsum)
        print('=======> Average PSNR = {0:.4f}, MSE = {1:.8f} * 10^{2}, MAE = {3:.8f} * 10^{4}'.format(psnr, *to_scientific(np.mean(msearr)), *to_scientific(np.mean(maearr))))
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            print('Test File Index: {0}, PSNR = {1:.4f}'.format(i, psnrs[i]))

if __name__ == '__main__':
    main()

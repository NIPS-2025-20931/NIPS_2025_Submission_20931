import sys
sys.path.append("../model")
from GAT import *
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
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

    coords = np.load(args.root + 'xy.npy')

    # The domain is defined on [0, 100] x [0, 100]
    coords = (coords / 100)

    radius = max_closest_dist(coords) * 1.5
    edge_index = All2AllPointCloudToGraph(coords, radius)

    batch_pos = torch.from_numpy(coords.astype(np.float32))
    batch_pos = batch_pos[None,...]
    batch_pos = batch_pos.to(device)

    edge_index = torch.from_numpy(edge_index.astype(np.int16)).t()
    edge_index = edge_index[None,...]
    edge_index = edge_index.to(device)

    num_nodes = batch_pos.shape[1]

    ####################################################################################################

    model = GAT_SR(in_features=in_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features, \
                        node_dim=1, latent_features=args.latent_feats, msg_passing_steps=args.msg_passing_steps, \
                        graph_nodes=num_nodes, pos_freq_const=10, pos_freq_num=pos_freq_num, \
                        pooling=args.latent_fuse)
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

    all_lf_smoke = all_lf_smoke[...,None].to(device)
    all_hf_smoke = all_hf_smoke[...,None].to(device)

    ####################################################################################################
    #test
    mask = np.load(args.root + 'mask.npy')
    mask_flatten = mask.flatten()
    selected_samples = np.arange(len(mask_flatten))[mask_flatten]

    psnrs = []
    msearr = []
    maearr = []
    model.eval()
    with torch.no_grad():
        errsum = 0
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            curpsnr = 0.0
            for j in range(80):
                lf_feats = all_lf_smoke[i*80+j][None,...]
                hf_feats = all_hf_smoke[i*80+j][None,...]
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
                fullres_arr = np.full(mask.shape[0]*mask.shape[1], np.nan)
                fullres_arr[selected_samples] = model_output
                plt.imshow(fullres_arr.reshape(mask.shape[0], mask.shape[1]).T, origin='lower', cmap='PRGn', vmin=0.0, vmax=1.0)
                plt.tight_layout()
                plt.axis('off')
                plt.savefig(args.dir_outputs + args.network_str + "_{0}_{1}.jpg".format(i, j+1), bbox_inches='tight', pad_inches=0)
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

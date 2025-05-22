import sys
sys.path.append("../model")
from FNO_DSE import *
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

def samplepoints(args, device, points, radius):
    index_arr = np.arange(len(points))
    points, reordered_index, npts = MultiResGraph(points=points.numpy(), index=index_arr, levels=3, pc2gfactor=3.0, g2gfactor=4.0, method='random')

    points = torch.from_numpy(points)

    g_points = points[None,...].to(device)
    transform = VFT(g_points[:,:,0], g_points[:,:,1], args.num_modes)

    return points, reordered_index, transform


@hydra.main(version_base=None, config_path="conf", config_name="FNO_DSE_test")
def main(args : DictConfig):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pytorch_device_config()

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

    grid_sample_points, reordered_index, grid_transform = samplepoints(args, device, pos_t, radius)

    ####################################################################################################

    model = FNO_DSE_SR(input_dim_edge=input_dim_edge, input_dim_node=input_dim_node, num_modes=args.num_modes, hidden_dim=args.latent_feats, output_dim_node=out_features)
    model.load_state_dict(torch.load(os.path.join(args.dir_weights, args.network_str + '_'+ str(args.start_epoch) + ".pth"), weights_only=True))
    model.to(device)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    size_all_kb = (param_size) / 1024
    print('Model size: {:.3f} KB'.format(size_all_kb))

    ####################################################################################################
    # Load all data
    all_lf_pressure = None
    all_hf_pressure = None
    for i in range(*dataidxrange[args.dataset]['test']):
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

    #test
    model.eval()
    psnrs = []
    msearr = []
    maearr = []
    with torch.no_grad():
        errsum = 0
        for i in range(dataidxrange[args.dataset]['test'][1]-dataidxrange[args.dataset]['test'][0]):
            curpsnr = 0.0
            for j in range(80):
                lf_feats = all_lf_pressure[i*80+j][None,...]
                hf_feats = all_hf_pressure[i*80+j][None,...]
                # ===================forward=====================
                model_output = model(lf_feats, grid_transform)
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
                fullres_arr = np.full(xyres[args.dataset][0]*xyres[args.dataset][1], np.nan)
                fullres_arr[selected_samples[reordered_index]] = model_output
                plt.imshow(fullres_arr.reshape(xyres[args.dataset][0], xyres[args.dataset][1]), origin='lower', cmap='RdBu', vmin=0.0, vmax=1.0)
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

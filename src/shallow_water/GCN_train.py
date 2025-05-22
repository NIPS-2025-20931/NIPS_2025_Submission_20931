import sys
sys.path.append("../model")
from GCN import *
from utils import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="GCN_train")
def main(args : DictConfig):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pytorch_device_config()

    data_indices = np.arange(0, dataidxrange[args.dataset]['train'][1]*12, dtype=int)

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
    batch_pos = batch_pos.repeat(args.batch_size, 1, 1)
    batch_pos = batch_pos.to(device)

    edge_index = torch.from_numpy(edge_index.astype(np.int16)).t()
    edge_index = edge_index[None,...]
    edge_index = edge_index.repeat(args.batch_size, 1, 1)
    edge_index = edge_index.to(device)

    num_samples = batch_pos.shape[1]

    ####################################################################################################
    model = GCN_SR_3D(in_features=in_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features, out_features=out_features, g_channel=1, \
                       g_layers=args.msg_passing_steps, g_features=args.g_feats, graph_nodes=num_samples, pos_freq_const=10, pos_freq_num=pos_freq_num, pooling=args.latent_fuse)
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
    all_lf_height = []
    all_hf_height = []
    for i in range(*dataidxrange[args.dataset]['train']):
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
    
    losses = []
    total_training_time_start = time.time()
    for epoch in tqdm(range(args.start_epoch, args.epochs)):
        print('epoch {0}'.format(epoch+1))
        total_loss = 0
        errsum = 0
        msearr = []
        tstart = time.time()
        np.random.shuffle(data_indices)
        for tidx in range(math.ceil(len(data_indices)/args.batch_size)):
            ts = data_indices[tidx*args.batch_size:(tidx+1)*args.batch_size]
            lf_feats = all_lf_height[ts]
            hf_feats = all_hf_height[ts]
            # ===================forward=====================
            model_output = model(batch_pos, lf_feats, edge_index)
            loss = criterion(model_output, hf_feats)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_mean_loss = loss.data.cpu().numpy()
            msearr.append(batch_mean_loss)
            errsum += batch_mean_loss * num_samples * args.batch_size
            total_loss += batch_mean_loss * args.batch_size
        tend = time.time()
        mse = errsum / (num_samples * len(data_indices))
        curr_psnr = 20. * np.log10(1.0) - 10. * np.log10(mse)
        losses.append(total_loss/len(data_indices))
        print('Training time: {0:.4f} for {1} data points x {2} timesteps, approx PSNR = {3:.4f}, approx MSE = {4:.8f} * 10^{5}'\
                  .format(tend-tstart, num_samples, len(data_indices), curr_psnr, *to_scientific(np.mean(msearr))))
        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(args.dir_weights, args.network_str + '_'+ str(epoch+1) + ".pth"))
    
    print('Total training time: {0:.4f}'.format(tend-total_training_time_start))
    

if __name__ == '__main__':
    main()

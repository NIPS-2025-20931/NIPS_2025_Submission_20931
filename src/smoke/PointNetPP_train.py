import sys
sys.path.append("../model")
from PointNetPP import *
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import math
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="conf", config_name="PointNetPP_train")
def main(args : DictConfig):
    # log hyperparameters
    print(args)

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pytorch_device_config()

    data_indices = np.arange(0, dataidxrange[args.dataset]['train'][1]*80, dtype=int)

    in_features = 2
    out_features = 1
    pos_freq_num = 30

    coords = np.load(args.root + 'xy.npy')

    # The domain is defined on [0, 100] x [0, 100]
    coords = (coords / 100)

    radius = max_closest_dist(coords) * 1.5

    pos_t = torch.from_numpy(coords.astype(np.float32))

    batch_pos = torch.from_numpy(coords.astype(np.float32))
    batch_pos = batch_pos[None,...]
    batch_pos = batch_pos.repeat(args.batch_size, 1, 1)
    batch_pos = batch_pos.to(device)

    zeros_column = torch.zeros(pos_t.shape[0], 1, dtype=pos_t.dtype, device=pos_t.device)
    pos_t = torch.cat((pos_t, zeros_column), dim=1)
    pos_t = pos_t.repeat(args.batch_size, 1, 1)
    pos_t = pos_t.to(device)

    num_nodes = batch_pos.shape[1]

    ####################################################################################################
    model = PointNetPP_SR(in_features=in_features, hidden_layers=args.hidden_layers, hidden_features=args.hidden_features, out_features=out_features, \
                            pc_channel=4, latent_feats=args.latent_feats, sampling_method=args.sampling_method)
    if args.start_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(args.dir_weights, args.network_str + '_'+ str(args.start_epoch) + ".pth")))
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
    all_lf_smoke = None
    all_hf_smoke = None
    for i in range(*dataidxrange[args.dataset]['train']):
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
            lf_feats = all_lf_smoke[ts]
            hf_feats = all_hf_smoke[ts]
            # ===================forward=====================
            model_output = model(batch_pos, pos_t.permute(0, 2, 1), lf_feats.permute(0, 2, 1))
            loss = criterion(model_output, hf_feats)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_mean_loss = loss.data.cpu().numpy()
            msearr.append(batch_mean_loss)
            errsum += batch_mean_loss * num_nodes * args.batch_size
            total_loss += batch_mean_loss * args.batch_size
        tend = time.time()
        mse = errsum / (num_nodes * len(data_indices))
        curr_psnr = 20. * np.log10(1.0) - 10. * np.log10(mse)
        losses.append(total_loss)
        print('Training time: {0:.4f} for {1} data points x {2} timesteps, approx PSNR = {3:.4f}, approx MSE = {4:.8f} * 10^{5}'\
                  .format(tend-tstart, num_nodes, len(data_indices), curr_psnr, *to_scientific(np.mean(msearr))))
        if (epoch+1) % args.check_every == 0:
            print("=> saving checkpoint at epoch {}".format(epoch + 1))
            torch.save(model.state_dict(), os.path.join(args.dir_weights, args.network_str + '_'+ str(epoch+1) + ".pth"))
    
    print('Total training time: {0:.4f}'.format(tend-total_training_time_start))
    
if __name__ == '__main__':
    main()

import torch
import argparse
import mmcv
import os
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange
from lib.masked_adam import MaskedAdam
from run_dvgo import load_existed_model
from lib import dvgo, dcvgo, dmpigo, utils

from run_dvgo import load_everything
from train_single_model import seed_everything
import copy
import time
import numpy as np
import imageio

import moe



def save_models(ensemble, basedir, exp_name, m_names):

    os.makedirs(os.path.join(basedir, exp_name), exist_ok=True)
    
    outpath_list = []
    for model, name in zip(ensemble.models, m_names):

        outpath = os.path.join(basedir, exp_name, f'{name}.tar')

        torch.save({
            'model_kwargs': model.get_kwargs(),
            'model_state_dict': model.state_dict(),
            'optim_state_dict': ensemble.optimizer.state_dict(),
            'global_step': 0,
        }, outpath)

        print("model saved at ", outpath)
        outpath_list.append(outpath)


    return outpath_list


def create_optimizer(moe, cfg_train, global_step, is_gate=False):

    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)

    param_group = []

    start = 0
    end = len(moe.models)
    
    # for each model
    if not is_gate:
        for i in range(start, end):

            for k in cfg_train.keys():
                if not k.startswith('lrate_'):
                    continue
                k = k[len('lrate_'):]

                if not hasattr(moe.models[i], k):
                    continue

                param = getattr(moe.models[i], k)
                if param is None:
                    print(f'create_optimizer_or_freeze_model: param {k} not exist')
                    continue

                lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
                if lr > 0:
                    print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
                    if isinstance(param, nn.Module):
                        param = param.parameters()
                    param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
                else:
                    print(f'create_optimizer_or_freeze_model: param {k} freeze')
                    param.requires_grad = False


    param_group.append({'params': moe.gate.k0.parameters(), 'lr': 1e-1, 'skip_zero_grad': True})
    param_group.append({'params': moe.gate.rgbnet.parameters(), 'lr': 1e-3, 'skip_zero_grad': False})


    return MaskedAdam(param_group)


def evaluate_ensemble(ensemble, args, cfg, data_dict, savedir=None):

    stepsize = cfg.fine_model_and_render.stepsize
    render_viewpoints_kwargs = {
        'ensemble': ensemble,
        'ndc': cfg.data.ndc,
        'render_kwargs': {
            'near': data_dict['near'],
            'far': data_dict['far'],
            'bg': 1 if cfg.data.white_bkgd else 0,
            'stepsize': stepsize,
            'inverse_y': cfg.data.inverse_y,
            'flip_x': cfg.data.flip_x,
            'flip_y': cfg.data.flip_y,
            'render_depth': True,
            }
        }
    
    return render_viewpoints(
            render_poses=data_dict['poses'][data_dict['i_test']],
            HW=data_dict['HW'][data_dict['i_test']],
            Ks=data_dict['Ks'][data_dict['i_test']],
            cfg=cfg,
            gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
            savedir=savedir, dump_images=args.dump_images,
            eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
            **render_viewpoints_kwargs)


@torch.no_grad()
def render_viewpoints(ensemble, render_poses, HW, Ks, ndc, cfg, render_kwargs,
                      gt_imgs=None, savedir=None, dump_images=False,
                      render_factor=0, render_video_flipy=False, render_video_rot90=0,
                      eval_ssim=False, eval_lpips_alex=False, eval_lpips_vgg=False, action='mean'):

    '''Render images for the given viewpoints; run evaluation if gt given.
    '''
    assert len(render_poses) == len(HW) and len(HW) == len(Ks)
    

    if render_factor!=0:
        HW = np.copy(HW)
        Ks = np.copy(Ks)
        HW = (HW/render_factor).astype(int)
        Ks[:, :2, :3] /= render_factor

    psnrs = []
    ssims = []
    lpips_alex = []
    lpips_vgg = []
    
    rgbs = []
    partial_maps = [ [] for j in range(ensemble.num_experts)]
    gate_maps = [[] for j in range(ensemble.num_experts)]
    
    ensemble.eval()
    
    mean_psnr, mean_ssim, mean_lpips_alex, mean_lpips_vgg = [-1, -1, -1, -1]
    for i, c2w in enumerate(tqdm(render_poses)):

        H, W = HW[i]
        K = Ks[i]
        c2w = torch.Tensor(c2w)
        rays_o, rays_d, viewdirs = dvgo.get_rays_of_a_view(
                H, W, K, c2w, ndc, inverse_y=render_kwargs['inverse_y'],
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        keys = ['rgb_marched', 'depth', 'alphainv_last']
        rays_o = rays_o.flatten(0,-2)
        rays_d = rays_d.flatten(0,-2)
        viewdirs = viewdirs.flatten(0,-2)

        rgb_list = []
             
        render_result_chunks = [
            ensemble(ro, rd, vd, **render_kwargs) for ro, rd, vd in zip(rays_o.split(8192, 0), rays_d.split(8192, 0), viewdirs.split(8192, 0))
        ]
        
        render_result = []
        for chunk_list in render_result_chunks:
            rgb_marched = 0
            for chunk in chunk_list:
                rgb_marched += chunk['rgb_marched']
            render_result.append(rgb_marched)

          
        del render_result_chunks
        torch.cuda.empty_cache()
        output = torch.cat(render_result).reshape(H, W, -1).detach().cpu().numpy()

        rgbs.append(output)
        
        if gt_imgs is not None and render_factor == 0:
            p = -10. * np.log10(np.mean(np.square(output - gt_imgs[i])))
            rgb_list.clear()
            

            psnrs.append(p)
            if eval_ssim:
                ss = utils.rgb_ssim(output, gt_imgs[i], max_val=1)
                ssims.append(ss)
            if eval_lpips_alex:
                lp = utils.rgb_lpips(output, gt_imgs[i], net_name='alex', device=c2w.device)
                lpips_alex.append(lp)
            if eval_lpips_vgg:
                lpips_vgg.append(utils.rgb_lpips(output, gt_imgs[i], net_name='vgg', device=c2w.device))


    if len(psnrs):
        mean = np.mean(psnrs)
        print('Testing psnr', mean, '(avg)')
        mean_psnr = np.mean(psnrs)
        
        if eval_ssim: 
            mean_ssim = np.mean(ssims)
            print('Testing ssim', mean_ssim, '(avg)')
        if eval_lpips_vgg: 
            mean_lpips_vgg = np.mean(lpips_vgg)
            print('Testing lpips (vgg)', mean_lpips_vgg, '(avg)')
        if eval_lpips_alex: 
            mean_lpips_alex = np.mean(lpips_alex)
            print('Testing lpips (alex)', mean_lpips_alex, '(avg)')


    if savedir is not None and dump_images:
        os.makedirs(os.path.join(savedir, 'out'), exist_ok=True)

        # for i in range(ensemble.num_experts):
        #     os.makedirs(os.path.join(savedir, f'partials_{i}'), exist_ok=True)
        #     if ensemble.use_gate is True:
        #         os.makedirs(os.path.join(savedir, f'gate_maps_{i}'), exist_ok=True)

        # os.makedirs(os.path.join(savedir, 'out'), exist_ok=True)

        for i in trange(len(rgbs)):
            rgb8 = utils.to8b(rgbs[i])
            filename = os.path.join(os.path.join(savedir, 'out'), '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)
            
        # for i in trange(len(partial_maps)):
        #     for j in trange(len(partial_maps[i])):
        #         rgb8 = utils.to8b(partial_maps[i][j])
        #         filename = os.path.join(os.path.join(savedir, f'partials_{i}'), '{:03d}.png'.format(j))
        #         imageio.imwrite(filename, rgb8)
        
        # if ensemble.use_gate is True:
        #     for i in trange(len(gate_maps)):
        #         for j in trange(len(gate_maps[i])):
        #             rgb8 = utils.to8b(gate_maps[i][j])
        #             filename = os.path.join(os.path.join(savedir, f'gate_maps_{i}'), '{:03d}.png'.format(j))
        #             imageio.imwrite(filename, rgb8)


    return mean_psnr, mean_ssim, mean_lpips_alex, mean_lpips_vgg


def train_moe(moe, cfg, cfg_model, cfg_train, data_dict, stage,
                   iters=20000, weights=None, alpha=0):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # data needed for training
    HW, Ks, near, far, i_train, i_val, i_test, poses, render_poses, images = [
        data_dict[k] for k in [
            'HW', 'Ks', 'near', 'far', 'i_train', 'i_val', 'i_test', 'poses', 'render_poses', 'images'
        ]
    ]


    optimizer = create_optimizer(moe, cfg_train, 0)
    moe.set_optimizer(optimizer)

    render_kwargs = {
        'near': data_dict['near'],
        'far': data_dict['far'],
        'bg': cfg.data.white_bkgd,
        'rand_bkgd': cfg.data.rand_bkgd,
        'stepsize': cfg_model.stepsize,
        'inverse_y': cfg.data.inverse_y,
        'flip_x': cfg.data.flip_x,
        'flip_y': cfg.data.flip_y,
    }


    def gather_training_rays():
        if data_dict['irregular_shape']:
            rgb_tr_ori = [images[i].to('cpu' if cfg.data.load2gpu_on_the_fly else device) for i in i_train]
        else:
            rgb_tr_ori = images[i_train].to('cpu' if cfg.data.load2gpu_on_the_fly else device)

        if cfg_train.ray_sampler == 'in_maskcache':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_in_maskcache_sampling(
                    rgb_tr_ori=rgb_tr_ori,
                    train_poses=poses[i_train],
                    HW=HW[i_train], Ks=Ks[i_train],
                    ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                    flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y,
                    model=moe.models[-1], render_kwargs=render_kwargs)
        elif cfg_train.ray_sampler == 'flatten':
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays_flatten(
                rgb_tr_ori=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        else:
            rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz = dvgo.get_training_rays(
                rgb_tr=rgb_tr_ori,
                train_poses=poses[i_train],
                HW=HW[i_train], Ks=Ks[i_train], ndc=cfg.data.ndc, inverse_y=cfg.data.inverse_y,
                flip_x=cfg.data.flip_x, flip_y=cfg.data.flip_y)
        index_generator = dvgo.batch_indices_generator(len(rgb_tr), cfg_train.N_rand)
        batch_index_sampler = lambda: next(index_generator)
        return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler

    # same for all models
    rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz, batch_index_sampler = gather_training_rays()


    torch.cuda.empty_cache()
    psnr_list = []
    global_step = -1


    for global_step in trange(1, 1 + iters):
        
        if cfg_train.ray_sampler in ['flatten', 'in_maskcache']:
            sel_i = batch_index_sampler()
            target = rgb_tr[sel_i]
            rays_o = rays_o_tr[sel_i]
            rays_d = rays_d_tr[sel_i]
            viewdirs = viewdirs_tr[sel_i]
        elif cfg_train.ray_sampler == 'random':
            sel_b = torch.randint(rgb_tr.shape[0], [cfg_train.N_rand])
            sel_r = torch.randint(rgb_tr.shape[1], [cfg_train.N_rand])
            sel_c = torch.randint(rgb_tr.shape[2], [cfg_train.N_rand])
            target = rgb_tr[sel_b, sel_r, sel_c]
            rays_o = rays_o_tr[sel_b, sel_r, sel_c]
            rays_d = rays_d_tr[sel_b, sel_r, sel_c]
            viewdirs = viewdirs_tr[sel_b, sel_r, sel_c]
        else:
            raise NotImplementedError
        
        if cfg.data.load2gpu_on_the_fly:
            target = target.to(device)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)
            viewdirs = viewdirs.to(device)

        # volume rendering
        render_result_list = moe(
                rays_o, rays_d, viewdirs,
                global_step=global_step,
                **render_kwargs)

        out = render_result_list[0]['rgb_marched']
        loss = F.mse_loss(out, target)
        

        optimizer.zero_grad(set_to_none=True)
        
        psnr = utils.mse2psnr(loss.detach())
        
        '''resolution-based aux loss'''
        m_s = render_result_list[0]['m_s']
        c_s = render_result_list[0]['c_s']
        aux_loss = alpha * (moe.num_experts / (sum(c_s)**2)) * sum([m_s[i] * c_s[i] * weights[i] for i in range(moe.num_experts)])

        # gradient descent
        loss = loss + aux_loss
        loss.backward()
        

        optimizer.step()
        
        psnr_list.append(psnr.item())

        # update lr
        decay_steps = cfg_train.lrate_decay * 1000
        decay_factor = 0.1 ** (1/decay_steps)
        
   
        for i_opt_g, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * decay_factor
        

        # check log
        if global_step % 500 == 0:
            to_write = f'scene_rep_reconstruction ({stage}): iter {global_step:6d} / Loss: {loss.item():.9f} /  PSNR: {np.mean(psnr_list):5.2f}'
            to_write += f' // aux_loss: {aux_loss.item():.9f} / c_s: {c_s} '
            tqdm.write(to_write)
            psnr_list = []


def config_parser():

    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", type=str)

    parser.add_argument('--ckpts', '--names-list', nargs='+', default=[],
                        help='paths to models for the ensemble')
    
    parser.add_argument('--renerf', action='store_true', default='',
                        help='use compressed models')

    parser.add_argument('--no_coarse', action='store_true')

    parser.add_argument("--seed", type=int, default=777,
                        help='Random seed')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--no_reload_optimizer", action='store_true',
                        help='do not reload optimizer state from saved ckpt')
    parser.add_argument("--ft_path", type=str, default='',
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--export_bbox_and_cams_only", type=str, default='',
                        help='export scene bbox and camera poses for debugging and 3d visualization')
    parser.add_argument("--export_coarse_only", type=str, default='')

    # testing options
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true')
    parser.add_argument("--render_train", action='store_true')
    parser.add_argument("--render_video", action='store_true')
    parser.add_argument("--render_video_flipy", action='store_true')
    parser.add_argument("--render_video_rot90", default=0, type=int)
    parser.add_argument("--render_video_factor", type=float, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--dump_images", action='store_true')
    parser.add_argument("--eval_ssim", action='store_true')
    parser.add_argument("--eval_lpips_alex", action='store_true')
    parser.add_argument("--eval_lpips_vgg", action='store_true')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=100000,
                        help='frequency of weight ckpt saving')

    # MoE options    
    parser.add_argument("--top_k", type=int, required=True, help="How many experts to use for each point (suggested: 1 or 2)")
    parser.add_argument("--num_experts", type=int, required=True, help="number of experts of the MoE")
    parser.add_argument("--dataset_name", type=str, required=True, help="dataset name (nerf_synthetic, nsvf, llff, TanksAndTemples etc...)")
    parser.add_argument("--scene", type=str, required=True, help="scene name (e.g., lego, mic, ship, etc..)")
    parser.add_argument("--resolutions", nargs='+', required=True, help="resolutions of models of the MoE e.g., 160 200 256 300")
    parser.add_argument("--alpha", type=int, help="value of alpha (lambda in the paper), weight of resolution-based aux loss", default=1e-3)
    parser.add_argument("--gate_res", type=int, help="resolution of gate (default 128)", default=128)
    parser.add_argument('--datadir', type=str, required=True, help='path of dataset dir')

    
    return parser


if __name__ == '__main__':
    
    parser = config_parser()
    args = parser.parse_args()
    args.render_test = True
    
    basedir = './'
    

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
         
            
    seed_everything(777)
    
    top_k = args.top_k
    num_experts = args.num_experts
    dataset_name = args.dataset_name
    scene = args.scene
    resolutions = sorted(args.resolutions[0].split(" "))
    alpha = args.alpha


    # res_penalty weights
    weights = [1, 1.5, 2.23, 3,35, 5]
  
        
    config_name = f'{scene}.py'
    config = os.path.join(basedir, 'configs', dataset_name, config_name)
    cfg = mmcv.Config.fromfile(config)
        

    # exp_name, m_names = get_exp_name(ckpts)
        
    if dataset_name == 'llff':
        model_class = moe.DirectMPIGOExpert
    else:
        model_class = moe.DirectVoxGOExpert

    models = []
    optimizers = []
    
    for res in resolutions:
        ckpt = os.path.join("./ckpts/", dataset_name, scene, f'{scene}_{res}', 'fine_last.tar')
        assert os.path.exists(ckpt), "You need to pre-train first the model at the specified resolutions. Look at train_single_model.py"
        model, optimizer, start = load_existed_model(args, cfg, cfg.fine_train, ckpt, model_class=model_class)
        models.append(model)
        optimizers.append(optimizer)
    
    
    gate = None

    if dataset_name == 'llff':
        ndc = True
        gate = moe.DirectMPIGOGate(xyz_min=models[-1].xyz_min, xyz_max=models[-1].xyz_max, 
                                    num_voxels=128**3, top_k=top_k, num_experts=num_experts,
                                    rgbnet_dim=9, rgbnet_width=64, use_dirs=True, mpi_depth=256, rgbnet_depth=0)
        gate.act_shift = models[-1].act_shift
    else:
        ndc = False
        gate = moe.DirectVoxGOGate(xyz_min=models[-1].xyz_min, xyz_max=models[-1].xyz_max, act_shift=models[-1].act_shift,
                                    num_voxels=128**3, top_k=top_k, num_experts=num_experts,
                                    alpha_init=models[-1].alpha_init, rgbnet_dim=12, rgbnet_width=64, use_dirs=True, rgbnet_depth=0)
            
    gate_ckpt = os.path.join(basedir, 'ckpts',f'{dataset_name}', scene, f'{scene}_{128}', 'fine_last.tar')
    
    '''we init the gate with the density and the feature grid of the lowest model (128^3). It helps the gate to converge better.'''
    gate.k0.grid = copy.deepcopy(models[0].k0.grid)
    gate.density.grid = copy.deepcopy(models[0].density.grid)


    '''we build the MoE'''
    models.pop(0) # the first model (128^3) is used only for initializing the gate
    moe = moe.DVGOMoE(models, gate, ndc=ndc, mask_cache=models[-1].mask_cache, density=models[-1].density)
    
    cfg.data.datadir = args.datadir

    data_dict = load_everything(cfg)
    
    resolutions.pop(0)
    exp_name = '+'.join(resolutions)
    
    if not args.render_only:

        iters = 30000 if dataset_name == 'llff' else 20000
        start = time.time()
        train_moe(moe, cfg, cfg.fine_model_and_render, cfg.fine_train,
                                data_dict, 'moe-training', iters=iters, alpha=alpha, weights=weights)
        training_time = time.time() - start
        state_dict_path_list = save_models(moe, os.path.join(basedir, 'moe', f'{dataset_name}', scene), exp_name, resolutions)
        
    
    mean_psnr, mean_ssim, mean_lpips_alex = 0, 0, 0
    
    if args.render_test:
        savedir = os.path.join(basedir, 'moe', dataset_name, scene, f'{exp_name}_topk={top_k}')
        os.makedirs(savedir, exist_ok=True)
        mean_psnr, mean_ssim, mean_lpips_alex, mean_lpips_vgg = evaluate_ensemble(moe, args, cfg, data_dict, savedir)
        

            
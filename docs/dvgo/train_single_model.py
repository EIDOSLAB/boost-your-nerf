'''
    This script is just a wrapper around dvgo, in order to train multiple models of different resolution faster
'''


from lib import dvgo, dbvgo, dcvgo, dmpigo, utils
import torch
import numpy as np
import mmcv
import os, time, random, argparse
from run_dvgo import load_everything, render_viewpoints, train



def seed_everything(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


  
def config_parser():
    
    '''Define command line arguments
    '''

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',
                        help='config file path')
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

    parser.add_argument("--i", type=int, default=0)
    
    parser.add_argument("--basedir", type=str, default=None, help="dir where you want to store your ckpts")
    parser.add_argument("--dataset_name", type=str, default=None, help="name of dataset, e.g, nerf, TanksAndTemples, nsvf, llff")
    parser.add_argument("--scene", type=str, default=None, help='name of the scene (e.g, lego, mic, ship)')
    parser.add_argument('--resolutions', nargs='+', required=True, help="resolutions of models, tipically [128, 160, 200, 256, 300, 350]")
    parser.add_argument('--datadir', type=str, required=True, help='path of dataset dir')

    return parser

    
if __name__ == '__main__':
    
    
    parser = config_parser()
    args = parser.parse_args()
    
    basedir = './' if args.basedir is None else args.basedir
    dataset_name = args.dataset_name
    scene = args.scene

    resolutions = args.resolutions[0].split(' ')


    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
            
    seed = 777        
    seed_everything(seed)

    
    for r in resolutions:
        
        args.config = os.path.join(basedir, 'configs', dataset_name, f'{scene}.py')
        
        cfg = mmcv.Config.fromfile(args.config)
        cfg.expname = f'{scene}_{r}'

        cfg.basedir = os.path.join(basedir, 'ckpts', f'{dataset_name}', scene)

        r = int(r)
        cfg.fine_model_and_render.num_voxels = r**3
        cfg.fine_model_and_render.num_voxels_base = r**3
        
        cfg.data.datadir = args.datadir
        
        if dataset_name == 'llff':
            cfg.fine_model_and_render.mpi_depth = r
        
        cfg.fine_train.N_iters = 1000 if dataset_name != 'llff' else 30000
        
        data_dict = load_everything(cfg)
        
        train_time = 0
        if not args.render_only:
            pretrained_path = None
            start = time.time()
            train(args, cfg, data_dict, pretrained_path)
            train_time = (time.time() - start) / 60
        if args.render_test or args.render_train or args.render_video:
            if args.ft_path:
                ckpt_path = args.ft_path
            else:
                ckpt_path = os.path.join(cfg.basedir, cfg.expname, 'fine_last.tar')
            ckpt_name = ckpt_path.split('/')[-1][:-4]
            if cfg.data.ndc:
                model_class = dmpigo.DirectMPIGO
            elif cfg.data.unbounded_inward:
                model_class = dcvgo.DirectContractedVoxGO
            else:
                model_class = dvgo.DirectVoxGO
                
            gate_class=None
            model = utils.load_model(model_class, ckpt_path).to(device)
            number_of_parameters = sum(p.numel() for p in model.parameters())   

            render_viewpoints_kwargs = {
                'model': model,
                'ndc': cfg.data.ndc,
                'render_kwargs': {
                    'near': data_dict['near'],
                    'far': data_dict['far'],
                    'bg': 1 if cfg.data.white_bkgd else 0,
                    'stepsize': 0.5,
                    'inverse_y': cfg.data.inverse_y,
                    'flip_x': cfg.data.flip_x,
                    'flip_y': cfg.data.flip_y,
                    'render_depth': True,
                },
                'cfg': cfg
            }
            
            if args.render_test:

                testsavedir = os.path.join(cfg.basedir, cfg.expname, 'render_test')
                os.makedirs(testsavedir, exist_ok=True)
                print('All results are dumped into', testsavedir)
                mean_psnr, mean_ssim, mean_lpips_vgg, mean_lpips_alex = render_viewpoints(
                        render_poses=data_dict['poses'][data_dict['i_test']],
                        HW=data_dict['HW'][data_dict['i_test']],
                        Ks=data_dict['Ks'][data_dict['i_test']],
                        gt_imgs=[data_dict['images'][i].cpu().numpy() for i in data_dict['i_test']],
                        savedir=testsavedir, dump_images=args.dump_images,
                        eval_ssim=args.eval_ssim, eval_lpips_alex=args.eval_lpips_alex, eval_lpips_vgg=args.eval_lpips_vgg,
                        **render_viewpoints_kwargs)
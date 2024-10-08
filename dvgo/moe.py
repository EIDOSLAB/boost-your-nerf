import os
import numpy as np

import torch
import torch.nn as nn

from torch_scatter import segment_coo


from lib import grid
from torch.utils.cpp_extension import load

from lib.dmpigo import DirectMPIGO
from lib.dvgo import Alphas2Weights, DirectVoxGO, Raw2Alpha
parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
        name='render_utils_cuda',
        sources=[
            os.path.join(parent_dir, 'lib', path)
            for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
        verbose=True)


'''Gate Model (based on DirectVoxGO)'''
class DirectVoxGOGate(nn.Module):
    def __init__(self, xyz_min, xyz_max, act_shift,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=12, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=0, rgbnet_width=128,
                 viewbase_pe=4, num_experts=3, top_k=2, use_dirs=True,
                 **kwargs):
        super().__init__()
        
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres
        
        self.num_experts = num_experts
        self.top_k = top_k if top_k is not None else num_experts

        # determine based grid resolution
        self.num_voxels_base = num_voxels
        self.voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        self.act_shift = act_shift
        self.use_dirs = use_dirs
        #self.register_buffer('act_shift', torch.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit

        self.k0_dim = rgbnet_dim
        self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
        self.rgbnet_direct = rgbnet_direct

     
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).to(self.k0.grid.device)
        
        dim0 = self.k0_dim
        
        if self.use_dirs:
            dim0 += (3+3*viewbase_pe*2)


        self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, num_experts),
            )
        nn.init.constant_(self.rgbnet[-1].bias, 0)
        print('dvgo: feature voxel grid', self.k0)
        print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)


    def forward(self, ray_pts, viewdirs=None, ray_id=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''


        rgb_logit = self.k0(ray_pts)

        if self.use_dirs:
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
            rgb_feat = torch.cat([rgb_logit, viewdirs_emb], -1)
            rgb_logit = rgb_feat
        
        rgb_logit = self.rgbnet(rgb_logit)

        return rgb_logit 
    
    
    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3)
        self.world_size = ((self.xyz_max - self.xyz_min) / self.voxel_size).long()
        self.voxel_size_ratio = self.voxel_size / self.voxel_size_base
        print('dvgo: voxel_size      ', self.voxel_size)
        print('dvgo: world_size      ', self.world_size)
        print('dvgo: voxel_size_base ', self.voxel_size_base)
        print('dvgo: voxel_size_ratio', self.voxel_size_ratio)
    
    
    
'''Gate Model (based on DirectMPIGO)'''  
class DirectMPIGOGate(nn.Module):
    def __init__(self, xyz_min, xyz_max, num_experts, top_k,
                 num_voxels=0, mpi_depth=128,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=0, rgbnet_width=64,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectMPIGO, self).__init__()
        self.register_buffer('xyz_min', torch.Tensor(xyz_min))
        self.register_buffer('xyz_max', torch.Tensor(xyz_max))
        self.fast_color_thres = fast_color_thres

        # determine init grid resolution
        self._set_grid_resolution(num_voxels, mpi_depth)
        
        self.top_k = top_k
        self.num_experts = num_experts

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init density bias so that the initial contribution (the alpha values)
        # of each query points on a ray is equal
        self.act_shift = grid.DenseGrid(
                channels=1, world_size=[1,1,mpi_depth],
                xyz_min=xyz_min, xyz_max=xyz_max)
        self.act_shift.grid.requires_grad = False
        with torch.no_grad():
            g = np.full([mpi_depth], 1./mpi_depth - 1e-6)
            p = [1-g[0]]
            for i in range(1, len(g)):
                p.append((1-g[:i+1].sum())/(1-g[:i].sum()))
            for i in range(len(p)):
                self.act_shift.grid[..., i].fill_(np.log(p[i] ** (-1/self.voxel_size_ratio) - 1))

        # init color representation
        # feature voxel grid + shallow MLP  (fine stage)
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        
        self.k0_type = k0_type
        self.k0_config = k0_config
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self.world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self.world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.register_buffer('viewfreq', torch.FloatTensor([(2**i) for i in range(viewbase_pe)]))
            dim0 = (3+3*viewbase_pe*2) + self.k0_dim
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(inplace=True),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU(inplace=True))
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, self.num_experts),
            )
            nn.init.constant_(self.rgbnet[-1].bias, 0)
            
        self.viewfreq = torch.FloatTensor([(2**i) for i in range(viewbase_pe)]).to(self.k0.grid.device)

        print('dmpigo: density grid', self.density)
        print('dmpigo: feature grid', self.k0)
        print('dmpigo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self.world_size
        if mask_cache_path is not None and mask_cache_path:
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            self_grid_xyz = torch.stack(torch.meshgrid(
                torch.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0]),
                torch.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1]),
                torch.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2]),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            mask = torch.ones(list(mask_cache_world_size), dtype=torch.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)


    def forward(self, ray_pts, viewdirs=None, ray_id=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''

        # query for color
        vox_emb = self.k0(ray_pts)

        # view-dependent color emission
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        viewdirs_emb = viewdirs_emb[ray_id]
        rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
        rgb_logit = self.rgbnet(rgb_feat)
            
        return rgb_logit


    def _set_grid_resolution(self, num_voxels, mpi_depth):
        # Determine grid resolution
        self.num_voxels = num_voxels
        self.mpi_depth = mpi_depth
        r = (num_voxels / self.mpi_depth / (self.xyz_max - self.xyz_min)[:2].prod()).sqrt()
        self.world_size = torch.zeros(3, dtype=torch.long)
        self.world_size[:2] = (self.xyz_max - self.xyz_min)[:2] * r
        self.world_size[2] = self.mpi_depth
        self.voxel_size_ratio = 256. / mpi_depth
        print('dmpigo: world_size      ', self.world_size)
        print('dmpigo: voxel_size_ratio', self.voxel_size_ratio)
    
    
    
'''Expert Model (based on DirectVoxGO)'''
class DirectVoxGOExpert(DirectVoxGO):
    
    
    def forward(self, rays_o, viewdirs, ray_pts, ray_id, interval, mask=None):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only support point queries in [N, 3] format'
        
        N = len(rays_o)
        
        ret_dict = {}
        
        if mask is not None:
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            

        # query for alpha w/ post-activation
        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)

        k0 = self.k0(ray_pts)

        # view-dependent color emission
        if self.rgbnet_direct:
            k0_view = k0
        else:
            k0_view = k0[:, 3:]
            
        viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
        viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
        viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
        rgb_feat = torch.cat([k0_view, viewdirs_emb], -1)
        rgb_logit = self.rgbnet(rgb_feat)
            

        radiance = torch.sigmoid(rgb_logit)

        ret_dict.update({
            'radiance': radiance,
            'alpha': alpha,
            'ray_id': ray_id,

        })

        return ret_dict
    
    
'''Expert Model (based on DirectMPIGO)'''
class DirectMPIGOExpert(DirectMPIGO):
    
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, mpi_depth=0,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=0,
                 **kwargs):
        super().__init__(xyz_min, xyz_max,
                 num_voxels, mpi_depth,
                 mask_cache_path, mask_cache_thres, mask_cache_world_size,
                 fast_color_thres,
                 density_type, k0_type,
                 density_config, k0_config,
                 rgbnet_dim,
                 rgbnet_depth, rgbnet_width,
                 viewbase_pe,
                 **kwargs)
        self.voxel_size = 0
    
    
    def forward(self, rays_o, rays_d, viewdirs, ray_pts, ray_id, step_id, interval, g_val=None, mask=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'

        ret_dict = {}

        # query for alpha w/ post-activation
        if mask is not None:
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            
        density = self.density(ray_pts) + self.act_shift(ray_pts)
        alpha = self.activate_density(density, interval)


        # query for color
        vox_emb = self.k0(ray_pts)

        if self.rgbnet is None:
            # no view-depend effect
            rgb = torch.sigmoid(vox_emb)
        else:
            # view-dependent color emission
            viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
            viewdirs_emb = torch.cat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
            viewdirs_emb = viewdirs_emb[ray_id]
            rgb_feat = torch.cat([vox_emb, viewdirs_emb], -1)
            rgb_logit = self.rgbnet(rgb_feat)
            radiance = torch.sigmoid(rgb_logit)

        ret_dict.update({
            'alpha': alpha,
            'raw_alpha': alpha,
            'radiance': radiance,
            'ray_id': ray_id,

        })


        return ret_dict
    



def sample_ray(rays_o, rays_d, voxel_size, xyz_min, xyz_max, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id


'''Model'''
class DVGOMoE(torch.nn.Module):
    
    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self.voxel_size_ratio
        shape = density.shape
        return Raw2Alpha.apply(density.flatten(), self.models[-1].act_shift, interval).reshape(shape)
        
    
    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        
        xyz_max = self.models[-1].xyz_max
        xyz_min = self.models[-1].xyz_min
        voxel_size = self.models[-1].voxel_size
        
        far = 1e9 
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        stepdist = stepsize * voxel_size
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils_cuda.sample_pts_on_rays(
            rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist)
        mask_inbbox = ~mask_outbbox
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        
        interval = stepsize * self.models[-1].voxel_size_ratio
        
        return ray_pts, ray_id, step_id, interval
    
    
    def __init__(self, models, gate=None, ndc=False, use_gate=True, mask_cache=None, density=None) -> None:
        super().__init__()
        self.models = nn.ModuleList([m for m in models])
        self.num_experts = len(models)
        self.mask_cache = mask_cache
        self.density = density
        self.fast_color_thres = 1e-4 if ndc == False else 7.8125e-4
        self.gate = gate
        
        self.act_shift = None
        if hasattr(models[-1], 'act_shift'):
            self.act_shift = models[-1].act_shift
        self.ndc = ndc



    def forward(self, rays_o, rays_d, viewdirs, global_step=3001, **render_kwargs):

        ret_dict = {}
        N = len(rays_o)

        # sample points
        ray_pts, ray_id, step_id, interval = self.sample_ray(rays_o=rays_o, rays_d=rays_d, 
                                                                voxel_size=self.models[-1].voxel_size, 
                                                                voxel_size_ratio = self.models[-1].voxel_size_ratio,
                                                                **render_kwargs)
        '''filter points'''
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            

        density = self.density(ray_pts)
        alpha = self.activate_density(density, interval)

        
        '''filter points (low density)'''
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        '''filter points (low transmittance)'''
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)
        if self.fast_color_thres > 0:
            mask2 = (weights > self.fast_color_thres)
            weights = weights[mask2]
            alpha = alpha[mask2]
            ray_pts = ray_pts[mask2]
            ray_id = ray_id[mask2]
            step_id = step_id[mask2] 
        
        r_s = torch.zeros(self.num_experts, *ray_pts.shape)
        alpha_s = torch.zeros(self.num_experts, ray_pts.shape[0])

        
        '''compute gate scores '''
        logits = self.gate(ray_pts, viewdirs, ray_id) 
        probs = torch.softmax(logits, dim=-1)
        top_k = self.gate.top_k

        vals, indices = torch.topk(probs, k=top_k)
        
        '''normalize weights so that they sum to 1 (only if K > 1)'''
        if self.gate.top_k > 1:    
            vals = vals / torch.sum(vals, dim=-1).unsqueeze(-1)
        
        c_s =  []
        m_s = []
        masks = []
        for i in range(self.num_experts):
    
            mask = indices == i
            g_val = (vals[mask]).unsqueeze(-1)
            mask = mask.sum(dim=-1).bool()
        
            render_result = self.models[i](rays_o, viewdirs, ray_pts, ray_id, interval,mask=mask)

            radiance = render_result['radiance']
            alpha = render_result['alpha']

            '''multiply by gate weights'''
            r_s[i][mask] = radiance * g_val
            alpha_s[i][mask] = alpha * g_val.squeeze()

            c_s.append(torch.sum(mask).item())
            m_s.append(torch.sum(probs[:, i]))
            masks.append(mask)


            ret_dict.update({'c_s': c_s, 'm_s': m_s })    

            
            # if self.training == False: 
            #     partial_maps = []
            #     gate_maps = []
            #     for i in range(self.num_experts):
                    
            #         alpha = torch.sum(alpha_s, dim=0)
            #         '''compute weights and alpha_inv'''
            #         w, a_inv = Alphas2Weights.apply(alpha_s[i], ray_id, N)
            #         p_out = segment_coo(
            #                 src=(w.unsqueeze(-1) * r_s[i]),
            #                 index=ray_id,
            #                 out=torch.zeros([N, 3]),
            #                 reduce='sum')
            #         p_out += (a_inv.unsqueeze(-1) * render_kwargs['bg'])
            #         partial_maps.append(p_out)
                
            #         gate_map = segment_coo(
            #                 src=(w * probs[:, i]),
            #                 index=ray_id,
            #                 out=torch.zeros([N]),
            #                 reduce='sum')
            #         #gate_map += (a_inv * render_kwargs['bg'])
            #         # gate_map[torch.sum(gate_map, dim=-1) == 1] = 0
            #         gate_maps.append(gate_map)
                
            #     ret_dict.update({'partial_maps': partial_maps, 'gate_maps': gate_maps})
                
            # r_s = torch.sum(r_s, dim=0) 

            
    # else:
            # for i, model in enumerate(self.models):
            #     render_result = model(
            #             rays_o, rays_d, viewdirs, ray_pts, ray_id, step_id, interval,
            #             global_step=global_step, is_train=True,
            #             **render_kwargs)

            #     radiance = render_result['radiance']
            #     alpha = render_result['alpha']

            #     r_s[i] = radiance 
            #     alpha_s[i] = alpha 
            

        # if self.training == False: 
        #     partial_maps = []
        #     for i in range(self.num_experts):
        #         w, a_inv = Alphas2Weights.apply(alpha_s[i], ray_id, N)
        #         p_out = segment_coo(
        #                     src=(w.unsqueeze(-1) * r_s[i]),
        #                     index=ray_id,
        #                     out=torch.zeros([N, 3]),
        #                     reduce='sum')
        #         p_out += (a_inv.unsqueeze(-1) * render_kwargs['bg'])
        #         partial_maps.append(p_out)
                
        #     ret_dict.update({'partial_maps': partial_maps})

        rgb = torch.sum(r_s, dim=0)
        alpha = torch.sum(alpha_s, dim=0)
        
        '''compute weights and alpha_inv'''
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N)

        rgb_marched = segment_coo(
            src=(weights.unsqueeze(-1) * rgb),
            index=ray_id,
            out=torch.zeros([N, 3]),
            reduce='sum')

        rgb_marched += (alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        
        ret_dict.update({'rgb_marched': rgb_marched})        
        
        return [ret_dict]
    

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

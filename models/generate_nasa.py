import sys
sys.path.append('/BS/garvita/work/code/if-net')

import data_processing.implicit_waterproofing as iw
import mcubes
import trimesh
import torch
import os
from glob import glob
import numpy as np
import torch.nn as nn

import ipdb
import sys
# sys.path.append('/BS/garvita/work/code/occupancy_networks/im2mesh/utils')
# from libmise import MISE
# from psbody.mesh import Mesh

class Generator(object):
    def __init__(self,  model_occ, threshold, exp_name, checkpoint = None, device = torch.device("cuda"), resolution = 16, batch_points = 1000000, root_dir='', body=False, skirt=False, model_type='D'):

        self.model_occ = model_occ.to(device)
        self.model_type = model_type

        self.model_occ.eval()
        self.threshold = threshold
        self.device = device
        self.resolution = resolution
        self.resolution = resolution
        self.checkpoint_path = root_dir + '/{}/checkpoints/'.format( exp_name)
        self.load_checkpoint(checkpoint)

        self.batch_points = batch_points

        self.min = -0.4
        self.max = 0.4

        self.min = -1.0
        self.max = 1.0

        grid_points = iw.create_grid_points_from_bounds(self.min, self.max, self.resolution)
        # grid_points[:, 0], grid_points[:, 2] = grid_points[:, 2], grid_points[:, 0].copy()
        #
        a = self.max + self.min
        b = self.max - self.min

        grid_coords = 2 * grid_points - a
        grid_coords = grid_coords / b
        self.global_scale = 1.0
        grid_coords = torch.from_numpy(grid_coords).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        self.grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
        self.out_actvn = nn.Sigmoid()
        self.n_part = 24
        self.skinning_wgt = torch.from_numpy(np.load('/BS/garvita2/static00/cloth_seq/upper_gar/skinning.npz')['skin']).to(self.device, dtype=torch.float)
        #load tpose mesh
        if body:
            self.skinning_wgt = torch.from_numpy(np.load('/BS/garvita2/static00/cloth_seq/upper_gar/skinning_body.npy')).to(self.device, dtype=torch.float)

        if skirt:
            skinning_body = np.load('/BS/garvita2/static00/cloth_seq/upper_gar/skinning_body.npy')
            skinning_skirt = np.load('/BS/cloth-anim/static00/tailor_data/skirt_weight.npz')['w']
            self.skinning_wgt = torch.from_numpy(np.matmul(skinning_skirt, skinning_body)).to(self.device, dtype=torch.float)
        tpose_path = '/BS/garvita4/static00/clothing_sdf/tailor_shirt/tpose.obj'
        self.body =False
        if body:
            self.body= True
            tpose_path = os.path.join('/BS/RVH_3dscan_raw2/static00/CAPE/cape_canonical/minimal_body_shape/03375.obj')
        if skirt:
            tpose_path = os.path.join('/BS/RVH_3dscan_raw2/static00/clothing_sdf/tailor_skirt/tpose.obj')

        mesh = trimesh.load(tpose_path, process=False)
        self.tpose_verts =  torch.from_numpy(mesh.vertices).to(self.device, dtype=torch.float).unsqueeze(0)
        self.upsampling_steps = 4
        self.padding = 0.25
        self.mise_res = resolution

        self.biject_id = [0, 1, 2, 3, 4,5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
        self.biject_id2 = [7,8, 10, 11, 20, 21, 22, 23]


    def tranform_pts(self, pts, transform, trans=None):
        if trans is not None:
            pts = pts - trans.unsqueeze(1)

        ones = torch.ones(pts.shape[0], pts.shape[1], 1).to(self.device)
        pts = torch.cat((pts, ones), dim=2)
        # transformed_pts = []
        # for i in range(self.n_part):
        #     transformed_pts.append(torch.matmul(pts, transform[:, i, :, :])[:,:,:3])
        # ipdb.set_trace()
        pts = pts.unsqueeze(1)
        pts = pts.repeat(1,self.n_part,1,1).permute(0,1,3,2)
        transformed_pts = torch.matmul(transform, pts)
        # #transformed_pts = torch.index_select(transformed_pts,2,self.vids)
        # transformed_pts = torch.stack([torch.sum(weights * transformed_pts[:, :, 0, :], dim=1),
        #                            torch.sum(weights * transformed_pts[:, :, 1, :], dim=1),
        #                            torch.sum(weights * transformed_pts[:, :, 2, :], dim=1)])
        #
        # # return torch.stack(transformed_pts).permute
        return transformed_pts

    def generate_mesh(self, data):
        device = self.device
        threshold =0.0
        box_size = 1 + self.padding
        #box_size get from min max  #todo: check this
        min = data['min'].detach().numpy()[0]
        max = data['max'].detach().numpy()[0]

        transform_inv = data["transform_inv"].to(device)
        smpl_verts =  data["smpl_verts"].to(device)
        gt_skin =  data["gt_skin"].to(device)
        trans_root = data['trans_root'].to(device)

        # split points to handle higher resolution
        grid_values = []
        all_pts = []
        all_logits = []
        logits_list = []
        grid_points = iw.create_grid_points_from_bounds(min -self.padding , max +self.padding, self.resolution)
        grid_coords = torch.from_numpy(grid_points).to(self.device, dtype=torch.float)
        grid_coords = torch.reshape(grid_coords, (1, len(grid_points), 3)).to(self.device)
        grid_points_split = torch.split(grid_coords, self.batch_points, dim=1)
        for pointsf in grid_points_split:
            all_pts.extend(pointsf[0].detach().cpu().numpy())
            with torch.no_grad():
                if self.model_type == 'U':
                    transformed_root = torch.reshape(trans_root, (trans_root.shape[0], self.n_part * 3)).unsqueeze(1).repeat(1, pointsf.shape[1], 1)
                    predictions = self.model_occ(pointsf, transformed_root)
                    logits = predictions[:,:,0]

                elif self.model_type == 'R':
                    transformed_pts = self.tranform_pts(pointsf, transform_inv)[:,:,:3, :]
                    _, predictions = self.model_occ(transformed_pts.permute(1,0,3,2))
                    logits = torch.max(predictions[:,:,:,0], dim=0)[0]
                else:

                    transformed_root = torch.reshape(trans_root, (trans_root.shape[0], self.n_part * 3)).unsqueeze(1).repeat(1, pointsf.shape[1], 1)
                    transformed_pts = self.tranform_pts(pointsf, transform_inv)[:, :, :3, :]
                    # todo : check the transformed points
                    _, predictions = self.model_occ(transformed_pts.permute(1, 0, 3, 2), transformed_root)
                    logits = torch.max(predictions[:,:,:, 0], dim=0)[0]


                #values = logits[0][0].cpu().detach().numpy()
            grid_values.append(logits[0])
        #     all_logits.extend(values.cpu().detach().numpy())
        #     logits_list.append((-1)*logits.squeeze(0).squeeze(0).detach().cpu())
        #
        # grid_values2 = torch.cat(grid_values, dim=0).cpu().detach().numpy()

        grid_values = torch.cat(grid_values, dim=0).cpu().detach().numpy()
        #logits = torch.cat(grid_values, dim=0).numpy()

        return  grid_values,  min, max
        #return grid_values2, grid_values, logits,  min, max



    def mesh_from_logits(self, logits, min, max):

        #logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        #logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        logits = np.reshape(logits, (self.resolution,) * 3)
        # padding to ba able to retrieve object close to bounding box bondary
        logits = np.pad(logits, ((1, 1), (1, 1), (1, 1)), 'constant', constant_values=0)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = 0.5
        #logits = (-1)*logits
        vertices, triangles = mcubes.marching_cubes(logits, threshold)

        # remove translation due to padding
        max = max + self.padding
        min = min - self.padding
        #vertices -= 1
        #rescale to original scale
        step = (max - min) / (self.mise_res )
        #step = (max - min) / (self.resolution*self.upsampling_steps - 1)
        vertices = np.multiply(vertices, step)
        vertices += [min, min,min]
        #vertices= vertices*self.global_scale
        return trimesh.Trimesh(vertices, triangles)

    def load_checkpoint(self, checkpoint):
        if checkpoint is None:
            checkpoints = glob(self.checkpoint_path+'/*')
            if len(checkpoints) == 0:
                print('No checkpoints found at {}'.format(self.checkpoint_path))

            checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
            checkpoints = np.array(checkpoints, dtype=int)
            checkpoints = np.sort(checkpoints)
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])
        else:
            path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoint)
        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model_occ.load_state_dict(checkpoint['model_state_occ_dict'])

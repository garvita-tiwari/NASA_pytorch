from __future__ import division
from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import imp
import trimesh
import torch
import ipdb
import pickle

class NasaData(Dataset):


    def __init__(self, mode, res = 32, data_path = 'shapenet/data/', split_file = 'shapenet/split.npz',
                 batch_size = 64, num_sample_points = 1024, num_workers = 12, sample_distribution = [1],
                 sample_sigmas = [0.015], num_parts=24, tailor=False, single=False, distinct=False, d_class='Jacket' , mesh_path =None,**kwargs):

        self.sample_distribution = np.array(sample_distribution)
        self.sample_sigmas = np.array(sample_sigmas)
        assert np.sum(self.sample_distribution) == 1
        assert np.any(self.sample_distribution < 0) == False
        assert len(self.sample_distribution) == len(self.sample_sigmas)

        self.path = data_path
        self.split = np.load(split_file)[mode]

        self.data = ['/{}'.format(self.split[i]) for i in range(len(self.split)) if
                     os.path.exists(os.path.join(data_path, '{}.npz'.format(self.split[i])))]
        #
        # self.data = ['/{:06}'.format(i) for i in range(900) if
        #              os.path.exists(os.path.join(data_path, '{:06}.npz'.format(i)))]
        self.res = res
        self.joints = [0, 1, 2, 3, 4,5, 6,7,8, 9, 10, 11,12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.n_part = len(self.joints)
        self.num_sample_points = num_sample_points
        self.batch_size = batch_size
        self.num_workers = num_workers
        # compute number of samples per sampling method
        self.num_samples = np.rint(self.sample_distribution * num_sample_points).astype(np.uint32)
        self.global_scale = 1.0
        self.tailor = tailor
        self.mode = mode
        self.d_class = d_class
        self.mesh_path = mesh_path

        self.skinning =np.load('/BS/garvita2/static00/cloth_seq/upper_gar/skinning_body.npy')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.path  + self.data[idx] +'.npz'

        nasa_data = np.load(path)
        trans_root =  nasa_data['trans_root']
        transform =  nasa_data['transform']
        all_labels = nasa_data['occ']
        transform_inv = np.array([np.linalg.inv(transform[i]) for i in range(24)])
        scale_data = nasa_data['bounds']
        bottom_cotner = scale_data[0]
        upper_corner = scale_data[1]
        min = np.min(bottom_cotner)
        max = np.max(upper_corner)
        name = self.data[idx]

        n_bbox = int(len(all_labels)/2)
        n_surf = int(len(all_labels)/2)

        #surface points
        boundary_sample_points0 = nasa_data['points'][:n_bbox] *self.global_scale
        boundary_sample_occupancies0 = nasa_data['occ'][:n_bbox]

        #110% box points
        boundary_sample_points1 = nasa_data['points'][n_bbox: 2*n_bbox] *self.global_scale
        boundary_sample_occupancies1 = nasa_data['occ'][n_bbox: 2*n_bbox]

        points = []
        occupancies = []

        num = int(self.num_sample_points/2)
        subsample_indices = np.random.randint(0, n_bbox, num)
        points.extend(boundary_sample_points0[subsample_indices])
        occupancies.extend(boundary_sample_occupancies0[subsample_indices])

        subsample_indices = np.random.randint(0, n_bbox, num)
        points.extend(boundary_sample_points1[subsample_indices])
        occupancies.extend(boundary_sample_occupancies1[subsample_indices])

        # todo: run this only for D and R models
        #load mesh file
        org_mesh = self.mesh_path  + self.data[idx] +'.obj'
        mesh = trimesh.load(org_mesh, process=False)
        verts = mesh.vertices
        subsample_indices = np.random.randint(0, len(verts), self.num_sample_points)

        #gt label for skinning
        #if this is slower, move this to training file
        smpl_verts = verts[subsample_indices]
        gt_skin = self.skinning[subsample_indices]
        assert len(points) == self.num_sample_points
        assert len(occupancies) == self.num_sample_points
        return {'path': self.data[idx],
                'gt_skin': np.array(gt_skin, dtype=np.float32),
                'transform_inv': np.array(transform_inv, dtype=np.float32),
                'smpl_verts': np.array(smpl_verts, dtype=np.float32),
                'points': np.array(points, dtype=np.float32),
                'label': np.array(occupancies, dtype=np.float32),
                'trans_root':  np.array(trans_root, dtype=np.float32),
                'min': min, 'max': max   }

    def get_loader(self, shuffle =True):

        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=True)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
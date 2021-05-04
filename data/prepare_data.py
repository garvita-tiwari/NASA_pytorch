import trimesh
import numpy as np

import argparse

import torch

from geo_utils import normalize_y_rotation
from psbody.mesh import Mesh, MeshViewer

import os
import pickle as pkl
import sys
sys.path.append('/BS/garvita/work/code/cloth_static/TailorNet')
sys.path.append('/BS/garvita/work/code/if-net/data_processing')
import implicit_waterproofing as iw
import ipdb

from models.torch_smpl4garment import TorchSMPL4Garment
ROOT = '/BS/garvita2/static00/cloth_seq/upper_gar/WIDC102'

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-frame', default=0 ,type=int)
    parser.add_argument('-beta_file', '--beta_file', default='/BS/cloth-anim/static00/tailor_data/shirt_male/shape/beta_000.npy',
                        type=str)
    parser.add_argument('-pose_file', '--pose_file', default='/BS/RVH/work/data/people_completion/poses/SMPL/female.pkl',
                        type=str)
    parser.add_argument('-out_dir', '--out_dir', default='/BS/cloth3d/static00/nasa_data/smpl_pose/train_data',
                        type=str)

    args = parser.parse_args()

    nasa_data_dir = args.out_dir
    if not os.path.exists(nasa_data_dir):
        os.makedirs(nasa_data_dir)

    pose_file = args.pose_file
    poses = pkl.load(open(pose_file, 'rb'), encoding="latin1")
    smpl_torch = TorchSMPL4Garment('female')
    sample_num = 1000000
    beta = np.load(args.beta_file)
    sub_id = '{:03}'.format(args.frame)

    sub_folder  =os.path.join(nasa_data_dir, sub_id)
    if not os.path.exists(sub_folder):
        os.makedirs(sub_folder)
    mesh_folder = os.path.join('/BS/cloth3d/static00/nasa_data/smpl_pose/meshes/{}'.format(sub_id))
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)
    print(len(poses))
    for j in range(len(poses)):
        theta_normalized = normalize_y_rotation(poses[j])
        frame_num = '{:06}'.format(j)
        out_file = os.path.join(sub_folder, frame_num +'.npz')
        if os.path.exists(out_file):
            print('already done:  ', out_file)
            continue

        #creat smpl
        pose_torch = torch.from_numpy(theta_normalized.astype(np.float32)).unsqueeze(0)
        betas_torch = torch.from_numpy(beta[:10].astype(np.float32)).unsqueeze(0)

        smpl_verts = smpl_torch.forward(pose_torch, betas_torch)
        transform = smpl_torch.A.detach().numpy()[0]
        transform_inv = np.array([np.linalg.inv(transform[i]) for i in range(24)])

        #rotate root joint using inverse transform
        joints  = smpl_torch.J_transformed.detach().numpy()[0]
        root_joint = np.array([joints[0, 0], joints[0, 1], joints[0, 2], 1])
        transformed_root = np.array([np.matmul(transform_inv[i], root_joint)[:3] for i in range(24) ])

        m1 = Mesh(v=smpl_verts.detach().numpy()[0], f=smpl_torch.faces)
        mesh =  trimesh.Trimesh(m1.v, smpl_torch.faces)

        #sample points on mesh surface and displace with sigma = 0.03
        boundary_points = []
        points = mesh.sample(sample_num)
        boundary_points_1 = points + 0.03 * np.random.randn(sample_num, 3)

        #sample points in bbox of 110%
        bottom_corner, upper_corner = mesh.bounds
        bottom_corner =  bottom_corner + 0.1*bottom_corner
        upper_corner =  upper_corner + 0.1*upper_corner
        x_pt = np.random.uniform(bottom_corner[0], upper_corner[0], sample_num)
        y_pt = np.random.uniform(bottom_corner[1], upper_corner[1], sample_num)
        z_pt = np.random.uniform(bottom_corner[2], upper_corner[2], sample_num)
        boundary_points_2 = np.concatenate([x_pt.reshape(-1,1), y_pt.reshape(-1,1), z_pt.reshape(-1,1)], axis=1)
        #check the smpl mesh and joint and unposed mesh here

        #find occupancy
        boundary_points.extend(boundary_points_1)
        boundary_points.extend(boundary_points_2)
        occupancies = iw.implicit_waterproofing(mesh, boundary_points)[0]

        #save the meshes to veriyfing
        m1.write_obj(os.path.join(mesh_folder, '{}.obj'.format(frame_num)))
        # m1 = Mesh(v=boundary_points_2[:10000])
        # m1.write_obj('/BS/garvita/work/code/tmp.obj')
        np.savez(out_file, bounds=np.array([bottom_corner, upper_corner]), points=boundary_points,
                 occ=occupancies, transform=transform, joint=joints, trans_root=transformed_root,  poses=poses[j])

        print('Finished {} {} '.format(sub_id, frame_num))



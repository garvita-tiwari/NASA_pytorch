import models.nasa_net as model
import data.load_data as data

import numpy as np
import argparse
import ipdb


import argparse
import torch

from generation_iterator import gen_iterator

parser = argparse.ArgumentParser(
    description='Run generation'
)

parser.add_argument('-d','--data' , default='WIDC102', type=str)
parser.add_argument('-dc','--data_class' , default='Jacket', type=str)
parser.add_argument('-dp','--data_path' , default='/BS/cloth3d/static00/nasa_data/smpl_pose/train_data/000', type=str)
parser.add_argument('-mp','--mesh_path' , default='/BS/cloth3d/static00/nasa_data/smpl_pose/meshes/000', type=str)
parser.add_argument('-rd','--root_dir' , default='/BS/RVH_3dscan_raw2/static00/model/nasa_smpl', type=str)
parser.add_argument('-split','--split_file' , default='/BS/cloth3d/static00/nasa_data/smpl_pose/split_test.npz', type=str)

parser.add_argument('-pose_enc', dest='pose_enc', action='store_true')
parser.set_defaults(pose_enc=False)
parser.add_argument('-feat_cat', dest='feat_cat', action='store_true')
parser.set_defaults(feat_cat=False)
parser.add_argument('-body_enc', dest='body_enc', action='store_true')
parser.set_defaults(body_enc=False)
parser.add_argument('-deform', dest='deform', action='store_true')
parser.set_defaults(deform=False)
parser.add_argument('-batch_size' , default=1, type=int)
parser.add_argument('-pose_feat' , default=8, type=int)
parser.add_argument('-xyz_feat' , default=16, type=int)
parser.add_argument('-num_part' , default=24, type=int)
parser.add_argument('-total_dim' , default=960, type=int)
parser.add_argument('-num_sample_points' , default=2048, type=int)
parser.add_argument('-res' , default=128, type=int)
parser.add_argument('-m','--models' , default='U', type=str)
parser.add_argument('-o','--optimizer' , default='Adam', type=str)
parser.add_argument('-l','--loss' , default='l2', type=str)
parser.add_argument('-lr','--learning_rate',default=0.0005, type=float)
parser.add_argument('-blend','--blending_weights',default=5., type=float)
parser.add_argument('-clamp_dist','--clamp_dist',default=2., type=float)

parser.add_argument('-lw','--label_w',default=0.5, type=float)
parser.add_argument('-mw','--minimal_w',default=0.05, type=float)
parser.add_argument('-mise', dest='mise', action='store_true')
parser.set_defaults(mise=False)
parser.add_argument('-mode' , default='test', type=str)
parser.add_argument('-ds','--data_split' , default='all', type=str)
parser.add_argument('-checkpoint', type=int)
parser.add_argument('-seq_id', default=0, type=int)
parser.add_argument('-retrieval_res' , default=256, type=int)
parser.add_argument('-batch_points', default=1000000, type=int)


try:
    args = parser.parse_args()
except:
    args = parser.parse_known_args()[0]

if args.models == 'U':
    net = model.NasaU(total_dim=args.total_dim, pose_enc=args.pose_enc, jts_freq=args.pose_feat, x_freq=args.xyz_feat)

elif args.models == 'R':
    net = model.NasaR(total_dim=args.total_dim, pose_enc=args.pose_enc, jts_freq=args.pose_feat, x_freq=args.xyz_feat)

elif args.models == 'D':
    net = model.NasaD(total_dim=args.total_dim, pose_enc=args.pose_enc, jts_freq=args.pose_feat, x_freq=args.xyz_feat, proj_dim=4)

else:
    print('wrong model type')

data_path = args.data_path
root_dir= args.root_dir

dataset = data.NasaData(args.mode, data_path = data_path, split_file= args.split_file, res=args.res,
                                   num_sample_points=args.num_sample_points, batch_size=args.batch_size, num_workers=30, d_class=args.data_class, mesh_path=args.mesh_path)

exp_name = 'NASA-{}-{}v-{}_m-{}_{}_{}_{}_{}'.format(args.num_sample_points, args.res, args.models, args.pose_enc,
                                                              args.loss, int(args.total_dim / args.num_part), args.label_w, args.minimal_w)


body_f = True

from models.generate_nasa import Generator

gen = Generator(net, 0.5, exp_name, checkpoint=args.checkpoint, resolution=args.retrieval_res,
                    batch_points=args.batch_points, root_dir=root_dir, body=body_f, model_type=args.models)

print('loading model........')
print('mode loaded........')

out_path = root_dir + '/{}/evaluation_{}_@{}/{}_mesh'.format(exp_name,args.checkpoint, args.retrieval_res, args.mode)


gen_iterator(out_path, dataset, gen)


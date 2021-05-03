import models.nasa_net as model
import data.load_data as data

import argparse
import torch
import ipdb
parser = argparse.ArgumentParser(
    description='Run Model'
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
parser.add_argument('-batch_size' , default=12, type=int)
parser.add_argument('-pose_feat' , default=8, type=int)
parser.add_argument('-xyz_feat' , default=16, type=int)
parser.add_argument('-num_part' , default=24, type=int)
parser.add_argument('-total_dim' , default=960, type=int)
parser.add_argument('-num_sample_points' , default=2048, type=int)
parser.add_argument('-res' , default=128, type=int)
parser.add_argument('-m','--models' , default='U', type=str)
parser.add_argument('-o','--optimizer' , default='Adam', type=str)
parser.add_argument('-l','--loss' , default='l2', type=str)
parser.add_argument('-lr','--learning_rate',default=0.0001, type=float)
parser.add_argument('-lw','--label_w',default=0.5, type=float)
parser.add_argument('-mw','--minimal_w',default=0.05, type=float)
parser.add_argument('-blend','--blending_weights',default=5., type=float)
parser.add_argument('-clamp_dist','--clamp_dist',default=2., type=float)


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

train_dataset = data.NasaData('train', data_path = data_path, split_file= args.split_file, res=args.res,
                                   num_sample_points=args.num_sample_points, batch_size=args.batch_size, num_workers=30, d_class=args.data_class, mesh_path=args.mesh_path)

val_dataset = data.NasaData('test', data_path = data_path, split_file= args.split_file, res=args.res,
                                 num_sample_points=args.num_sample_points, batch_size=args.batch_size, num_workers=30, d_class=args.data_class, mesh_path=args.mesh_path)

exp_name = 'NASA-{}-{}v-{}_m-{}_{}_{}_{}_{}'.format(args.num_sample_points, args.res, args.models, args.pose_enc,
                                                              args.loss, int(args.total_dim / args.num_part), args.label_w, args.minimal_w)

from models import train_nasa

print(exp_name)
print(data_path)



trainer = train_nasa.Trainer(net, torch.device("cuda"), train_dataset, val_dataset, exp_name,
                           optimizer=args.optimizer, loss_type=args.loss,
                           batch_size=args.batch_size, root_dir=root_dir, clamp_dist=args.clamp_dist, model_type=args.models, label_w=args.label_w, minimal_w=args.minimal_w)

trainer.train_model(20001)
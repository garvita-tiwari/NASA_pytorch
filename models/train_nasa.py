"""This is for training tailornet meshes"""
from __future__ import division
import torch
import torch.optim as optim
from torch.nn import functional as F
import os
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import numpy as np
import torch.nn as nn

import ipdb


class Trainer(object):

    def __init__(self, model_occ, device, train_dataset, val_dataset, exp_name, root_dir='', optimizer='Adam', loss_type='cross_entropy',  z_vec=False, batch_size=8, need_transform=False, use_joint=True, clamp_dist=2.0, model_type='D', label_w=0.5, minimal_w= 0.05):

        self.model_occ = model_occ.to(device)
        self.device = device

        if optimizer == 'Adam':
            self.optimizer_occ = optim.Adam(self.model_occ.parameters(), lr=1e-4)
        if optimizer == 'Adadelta':
            self.optimizer_occ = optim.Adadelta(self.model_occ.parameters())
        if optimizer == 'RMSprop':
            self.optimizer_occ = optim.RMSprop(self.model_occ.parameters(), momentum=0.9)


        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        #self.exp_path = os.path.dirname(__file__) + '/../experiments/{}/'.format( exp_name)
        self.exp_path = '{}/{}/'.format( root_dir, exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'.format( exp_name)
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary'.format(exp_name))
        self.val_min = None
        self.train_min = None
        self.loss = loss_type
        self.n_part = 24
        self.loss_mse = torch.nn.MSELoss()

        self.model_type = model_type

        self.batch_size= batch_size
        self.joints = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.joints = range(self.n_part)

        skinning_temp = np.load('/BS/garvita2/static00/cloth_seq/upper_gar/skinning.npz')['skin']
        skinning_temp = skinning_temp[:, self.joints]
        tmp2= np.zeros_like(skinning_temp)
        part_label = np.argmax(skinning_temp,axis=1)

        for i in range(tmp2.shape[0]):
            tmp2[i, part_label[i]] = 0.5
        self.part_label = torch.from_numpy(part_label.astype(np.int64)).cuda()
        #todo: change this to classification
        self.skinning_temp = torch.from_numpy(skinning_temp.astype(np.float32)).cuda()
        self.skinning_temp = self.skinning_temp.permute(1,0).unsqueeze(1).unsqueeze(1)
        self.skinning_temp = self.skinning_temp.repeat(1,self.batch_size,1,1)
        self.skinning2 = torch.from_numpy(tmp2.astype(np.float32)).cuda()
        self.skinning2 = self.skinning2.permute(1,0).unsqueeze(1).unsqueeze(1)
        self.skinning2 = self.skinning2.repeat(1,self.batch_size,1,1)
        self.vids = torch.tensor([0, 1, 2]).cuda()


        self.need_transform = need_transform
        self.use_joint = use_joint
        self.out_act = nn.Sigmoid()

        if loss_type == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif loss_type == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
        self.clamp_dist = clamp_dist

        self.label_w = label_w
        self.minimal_w  = minimal_w


    def train_step(self,batch, ep=None):

        self.model_occ.train()
        self.optimizer_occ.zero_grad()
        loss = self.compute_loss(batch, ep)
        loss.backward()
        self.optimizer_occ.step()

        return loss.item()

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

    def compute_loss(self,batch,ep=None):
        device = self.device
        occ_gt = batch.get('label').to(device)
        pts = batch.get("points").to(device)
        if self.model_type == 'U':
            transformed_root = batch.get('trans_root').to(device)
            transformed_root = torch.reshape(transformed_root, (transformed_root.shape[0], self.n_part * 3)).unsqueeze(1).repeat(1, pts.shape[1], 1)
            predictions = self.model_occ(pts, transformed_root)[:,:,0]
            total_loss = self.loss_l1(predictions, occ_gt)
        else:
            transform_inv = batch.get("transform_inv").to(device)
            smpl_verts = batch.get("smpl_verts").to(device)
            gt_skin = batch.get("gt_skin").to(device)
            gt_label =  torch.argmax(gt_skin,dim=2)
            one_hot = torch.nn.functional.one_hot(gt_label.to(torch.int64), self.n_part)
            transformed_pts = self.tranform_pts(pts, transform_inv)[:,:,:3, :]
            transformed_smpl = self.tranform_pts(smpl_verts, transform_inv)[:,:,:3, :]
            if self.model_type == 'R':
                predictions, parts = self.model_occ(transformed_pts.permute(1,0,3,2))
                _, pred_labels = self.model_occ(transformed_smpl.permute(1, 0, 3, 2))  # todo: dry run validity

            elif self.model_type == 'D':
                transformed_root = batch.get('trans_root').to(device)
                transformed_root = torch.reshape(transformed_root, (transformed_root.shape[0], self.n_part * 3)).unsqueeze(1).repeat(1, pts.shape[1], 1)
                predictions, parts = self.model_occ(transformed_pts.permute(1, 0, 3, 2), transformed_root)
                _, pred_labels = self.model_occ(transformed_smpl.permute(1, 0, 3, 2), transformed_root)  # todo: dry run validity

            predictions = predictions[:, :, 0]
            pred_labels = pred_labels[:,:,:,0].permute(1,2,0)
            loss_i = self.loss_l1(predictions, occ_gt)
            total_loss = loss_i
            if self.label_w != 0.0:
                label_loss = self.loss_l1(pred_labels, 0.5*one_hot.to(torch.float32))
                total_loss =  total_loss + self.label_w*label_loss
            if self.minimal_w != 0.0:
                minimal_loss = torch.mean(torch.square(parts[:,:,int(parts.shape[2]/2):,:]))
                total_loss =  total_loss + self.minimal_w*minimal_loss

            # todo: add minimal loss term

        return total_loss

    def train_model(self, epochs):
        loss = 0
        start = self.load_checkpoint()

        for epoch in range(start, epochs):
            sum_loss = 0
            print('Start epoch {}'.format(epoch))
            train_data_loader = self.train_dataset.get_loader()

            if epoch % 100 == 0:
                self.save_checkpoint(epoch)
                # self.remove_extra_ckpt(epoch)

            for batch in train_data_loader:
                loss = self.train_step(batch)
                print("Current loss: {}".format(loss))
                sum_loss += loss
            batch_loss = sum_loss / len(train_data_loader)
            if self.train_min is None:
                self.train_min = batch_loss
            if batch_loss < self.train_min:
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'train_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'train_min={}'.format(epoch), [epoch, batch_loss])

            val_loss = self.compute_val_loss()

            if self.val_min is None:
                self.val_min = val_loss

            if val_loss < self.val_min:
                self.val_min = val_loss
                self.save_checkpoint(epoch)
                for path in glob(self.exp_path + 'val_min=*'):
                    os.remove(path)
                np.save(self.exp_path + 'val_min={}'.format(epoch), [epoch, batch_loss])

            self.writer.add_scalar('training loss last batch', loss, epoch)
            self.writer.add_scalar('training loss batch avg', batch_loss, epoch)
            self.writer.add_scalar('val loss batch avg', val_loss, epoch)

    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(epoch)
        if not os.path.exists(path):

            torch.save({'epoch':epoch, 'model_state_occ_dict': self.model_occ.state_dict(),
                        'optimizer_occ_state_dict': self.optimizer_occ.state_dict()}, path,  _use_new_zipfile_serialization=False)

    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path+'/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        checkpoints = [os.path.splitext(os.path.basename(path))[0][17:] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=int)
        checkpoints = np.sort(checkpoints)
        path = self.checkpoint_path + 'checkpoint_epoch_{}.tar'.format(checkpoints[-1])

        print('Loaded checkpoint from: {}'.format(path))
        checkpoint = torch.load(path)
        self.model_occ.load_state_dict(checkpoint['model_state_occ_dict'])
        self.optimizer_occ.load_state_dict(checkpoint['optimizer_occ_state_dict'])
        epoch = checkpoint['epoch']
        return epoch

    def compute_val_loss(self):

        self.model_occ.eval()

        sum_val_loss = 0
        num_batches = 15
        for _ in range(num_batches):
            try:
                val_batch = self.val_data_iterator.next()
            except:
                self.val_data_iterator = self.val_dataset.get_loader().__iter__()
                val_batch = self.val_data_iterator.next()

            sum_val_loss += self.compute_loss( val_batch).item()

        return sum_val_loss / num_batches
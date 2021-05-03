import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
device= torch.device("cuda")

# NASA network modules- > Unstructured, Rigid, Defomable ---------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


class Fourier(nn.Module):
    def __init__(self, nmb=128, scale=10, ip=3):
        super(Fourier, self).__init__()
        self.b = torch.randn(ip, nmb)*scale
        self.b = self.b.to(device)
        self.pi = 3.14159265359
    def forward(self, v):
        x_proj = torch.matmul(2*self.pi*v, self.b)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

class PositionalEncoder():
    def __init__(self, number_frequencies, include_identity):
        freq_bands = torch.pow(2, torch.linspace(0., number_frequencies - 1, number_frequencies))
        self.embed_fns = []
        self.output_dim = 0
        self.number_frequencies = number_frequencies
        self.include_identity = include_identity
        if include_identity:
            self.embed_fns.append(lambda x: x)
            self.output_dim += 1
        if number_frequencies > 0:
            for freq in freq_bands:
                for periodic_fn in [torch.sin, torch.cos]:
                    self.embed_fns.append(lambda x, periodic_fn=periodic_fn, freq=freq: periodic_fn(x * freq))
                    self.output_dim += 1

    def encode(self, coordinate):
        return torch.cat([fn(coordinate) for fn in self.embed_fns], -1)


class NasaU(nn.Module):

    def __init__(self, total_dim=960, num_parts=24, pose_enc=False, jts_freq=8, x_freq=16, num_layers=5):
        super(NasaU, self).__init__()
        self.num_neuron = total_dim
        self.num_layers = num_layers
        self.num_parts = num_parts
        x_freq = x_freq
        jts_freq = jts_freq

        self.input_dim = 3 + self.num_parts * 3
        self.layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + 72 * 2 * jts_freq #todo: check this

            self.x_enc = PositionalEncoder(x_freq, True)
            self.jts_enc = PositionalEncoder(jts_freq, True)

        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 1))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))

        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, jts):

        batch_size = x.shape[0]
        num_pts = x.shape[2]
        if self.pose_enc:  #todo : check this
            x = x.permute(0, 2, 1)  # b, NUM, 3
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

            # reshape the input
            jts = jts.permute(0, 2, 1)  # b, NUM, 3
            jts = jts.reshape(jts.shape[0] * jts.shape[1], jts.shape[2])

            x = self.x_enc.encode(x)
            x = x.reshape(batch_size, num_pts, x.shape[1])
            x = x.permute(0, 2, 1)
            jts = self.jts_enc.encode(jts)
            jts = jts.reshape(batch_size, num_pts, jts.shape[1])
            jts = jts.permute(0, 2, 1)

        for i in range(self.num_layers - 1):
            if i == 0:
                x_net = torch.cat((x, jts), dim=2)
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net

        x_net = self.out_actvn(self.layers[-1](x_net))
        return x_net

class PartNetR(nn.Module):
    def __init__(self, total_dim=40, pose_enc=False, x_freq=16, num_layers=5, input_dim=3):
        super(PartNetR, self).__init__()
        self.num_neuron = total_dim
        self.num_layers = num_layers
        x_freq = x_freq
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 +  3 * 2 * x_freq #todo: check this
            self.x_enc = PositionalEncoder(x_freq, True)

        ##### create network
        current_dim = self.input_dim
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(current_dim, self.num_neuron))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
            current_dim = self.num_neuron
        self.layers.append(nn.Linear(current_dim, 1))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))
        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()  #todio: here ???


    def forward(self, x_net):
        for i in range(self.num_layers - 1):
            if i == 0:
                x_net = self.actvn(self.layers[i](x_net))
                residual = x_net
            else:
                x_net = self.actvn(self.layers[i](x_net) + residual)
                residual = x_net

        #x_net = self.out_actvn(self.layers[-1](x_net))
        x_net = self.layers[-1](x_net)
        return x_net


class NasaR(nn.Module):

    def __init__(self, total_dim=960, num_parts=24, pose_enc=False, jts_freq=8, x_freq=16, num_layers=5):
        super(NasaR, self).__init__()
        self.num_neuron = int(total_dim/num_parts)
        self.num_layers = num_layers
        self.num_parts = num_parts
        self.x_freq = x_freq
        self.jts_freq = jts_freq

        self.input_dim = 3 + self.num_parts * 3
        self.layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + 72 * 2 * jts_freq #todo: check this

            self.x_enc = PositionalEncoder(x_freq, True)
            self.jts_enc = PositionalEncoder(jts_freq, True)

        ##### create network
        for _ in range(self.num_parts ):
            self.layers.append(PartNetR(self.num_neuron, pose_enc=self.pose_enc, x_freq=self.x_freq))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
        #self.layers.append(nn.Linear(self.num_neuron, 1))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))
        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.soft_blend = 5.0 #todo: value from NASA

    def forward(self, x_all):
        all_parts = []

        for i in range(self.num_parts):

            x = x_all[i]
            batch_size = x.shape[0]
            num_pts = x.shape[2]
            if self.pose_enc:  #todo : check this
                x = x.permute(0, 2, 1)  # b, NUM, 3
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                x = self.x_enc.encode(x)
                x = x.reshape(batch_size, num_pts, x.shape[1])
                x = x.permute(0, 2, 1)

            all_parts.append(self.layers[i](x))
        #softmax

        all_parts = self.out_actvn(torch.stack(all_parts))

        weights = self.softmax(self.soft_blend*all_parts)
        return torch.sum(weights*all_parts, dim =0), all_parts



class NasaD(nn.Module):

    def __init__(self, total_dim=960, num_parts=24, pose_enc=False, jts_freq=8, x_freq=16, num_layers=5, proj_dim=4):
        super(NasaD, self).__init__()
        self.num_neuron = int(total_dim/num_parts)
        self.num_layers = num_layers
        self.num_parts = num_parts
        self.x_freq = x_freq
        self.jts_freq = jts_freq

        self.input_dim = 3 + proj_dim
        self.layers = nn.ModuleList()
        self.proj_layers = nn.ModuleList()
        self.pose_enc = pose_enc

        ### apply positional encoding on input features
        if self.pose_enc:
            self.input_dim = 3 + self.num_parts * 3 + 3 * 2 * x_freq + 72 * 2 * jts_freq #todo: check this

            self.x_enc = PositionalEncoder(x_freq, True)
            self.jts_enc = PositionalEncoder(jts_freq, True)

        ##### create network
        for _ in range(self.num_parts ):
            self.layers.append(PartNetR(self.num_neuron, pose_enc=self.pose_enc, x_freq=self.x_freq, input_dim=self.input_dim))
            self.proj_layers.append(nn.Linear(self.num_parts * 3, proj_dim ))
            #self.layers.append(nn.Conv1d(current_dim, self.num_neuron, 1))
        #self.layers.append(nn.Linear(self.num_neuron, 1))
        #self.layers.append(nn.Conv1d(current_dim, 1, 1))
        self.actvn = nn.LeakyReLU(0.1)
        self.out_actvn = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)
        self.soft_blend = 5.0 #todo: value from NASA

    def forward(self, x_all, jts):
        all_parts = []

        for i in range(self.num_parts):

            x = x_all[i]
            batch_size = x.shape[0]
            num_pts = x.shape[2]
            if self.pose_enc:  #todo : check this
                x = x.permute(0, 2, 1)  # b, NUM, 3
                x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
                x = self.x_enc.encode(x)
                x = x.reshape(batch_size, num_pts, x.shape[1])
                x = x.permute(0, 2, 1)
            #apply projection
            inp_proj = self.proj_layers[i](jts)
            x_net = torch.cat((x, inp_proj), dim=2)
            all_parts.append(self.layers[i](x_net))
        #softmax

        all_parts = self.out_actvn(torch.stack(all_parts))

        weights = self.softmax(self.soft_blend*all_parts)
        return torch.sum(weights*all_parts, dim =0), all_parts



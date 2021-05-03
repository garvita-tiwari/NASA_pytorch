import numpy as np
import ipdb
import os

import global_var
if __name__ == '__main__':
    nasa_data_dir = '/BS/cloth3d/static00/nasa_data/smpl_pose'

    train = []
    val = []
    for i  in range(892):
        # read the original mesh
        frame_name = '{:06}'.format(i)

        if i%10 == 0:
            val.append(frame_name)
        else:
            train.append(frame_name)

    split_file = '{}/split_test.npz'.format(global_var.nasa_data_dir)
    ipdb.set_trace()
    np.savez(split_file,train=train, test=val)

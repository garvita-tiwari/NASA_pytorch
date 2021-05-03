import os
import trimesh
import traceback
import pickle as pkl
import numpy as np
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import Pool


# def gen_iterator(out_path, dataset, gen_p , buff_p, start,end):
def gen_iterator(out_path, dataset, gen_p):

    global gen
    gen = gen_p

    if not os.path.exists(out_path):
        os.makedirs(out_path)
    print(out_path)


    # can be run on multiple machines: dataset is shuffled and already generated objects are skipped.
    loader = dataset.get_loader(shuffle=True)

    data_tupels = []
    for i, data in tqdm(enumerate(loader)):

        path = os.path.normpath(data['path'][0])
        # print(out_path, path)

        if os.path.exists(out_path + data['path'][0] +'.obj'):
            print('Path exists - skip! {}'.format(out_path + data['path'][0] +'.obj'))
            continue

        try:
            if len(data_tupels) > 20:
                create_meshes(data_tupels)
                data_tupels = []
            logits, min, max = gen.generate_mesh(data)
            data_tupels.append((logits,min, max, data, out_path))


        except Exception as err:
            print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))

    try:

        create_meshes(data_tupels)
        data_tupels = []
        logits,  min, max = gen.generate_mesh(data)
        data_tupels.append((logits,  min, max, data, out_path))


    except Exception as err:
        print('Error with {}: {}'.format(data['path'][0], traceback.format_exc()))


def save_mesh(data_tupel):
    logits,  min, max , data, out_path = data_tupel

    mesh = gen.mesh_from_logits(logits, min, max )

    # path = os.path.normpath(data['path'][0])
    # export_path = out_path + '/generation/{}/{}/'.format(path.split(os.sep)[-2], path.split(os.sep)[-1])

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mesh.export(out_path + data['path'][0] +'.obj')

def create_meshes(data_tupels):
    p = Pool(mp.cpu_count())
    p.map(save_mesh, data_tupels)
    p.close()
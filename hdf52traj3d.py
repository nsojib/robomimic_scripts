import os
import json
import h5py
import numpy as np

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import imageio
import tqdm
from robomimic.utils.file_utils import create_hdf5_filter_key
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import plotly.graph_objects as go
import numpy as np
import argparse


def save_as_3d_traj(ees, fullpath, width=800, height=800): 
    x,y,z = ees[:,0], ees[:,1], ees[:,2]
    mins=np.min(ees, axis=0)
    maxs=np.max(ees, axis=0)
    dmax=np.max( maxs-mins)

    fig = go.Figure(data=[go.Scatter3d(x=x-mins[0], y=y-mins[1], z=z-mins[2],
                                          mode='markers')])
    offset=0.1
    x_lim=dict(range=[-offset, dmax+offset])
    y_lim=dict(range=[-offset, dmax+offset])
    z_lim=dict(range=[-offset, dmax+offset])

    fig.update_layout(
                scene=dict(xaxis=x_lim, yaxis=y_lim, zaxis=z_lim),
                      margin=dict(l=0, r=0, b=0, t=0),
                width=width, 
                height=height 
        )
    fig.write_image(f"{fullpath}")

def main(dataset_path, save_dir): 
    if save_dir==None:
        save_dir = os.path.join(os.path.dirname(dataset_path), "action3d")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())

    lengths=[]
    for demo_name in demos:
        demo=f['data'][demo_name]
        num_samples=demo.attrs['num_samples']
        lengths.append(num_samples)

    lengths=np.array(lengths)

    print('Number of demos: ', len(demos))
    print('Max length: ', np.max(lengths))
    print('Min length: ', np.min(lengths))
    print('Mean length: ', np.mean(lengths))
 

    for demo_name in tqdm.tqdm(demos):
        obs_ee=f['data'][demo_name]['obs']['robot0_eef_pos']
        ees = np.array(obs_ee)[:, :3]
        save_as_3d_traj(ees, f"{save_dir}/{demo_name}.png")
        

if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, help='path to input hdf5 dataset', required=True)
    args.add_argument('--save_dir', type=str, help='path to save directory') 
    args = args.parse_args()
    main(args.dataset, args.save_dir)

# python hdf52traj3d.py --dataset /home/ns/robosuite/collects/1705874644_525442/demo101_jan21_image.hdf5

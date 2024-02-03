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
import argparse

np.set_printoptions(precision=4, suppress=True)


def main(dataset_path):
    """
    Normalize actions in a dataset to [-1,1] using mins and maxs
    Save mins and maxs as another key in the dataset
    """
    
    #TODO: check if dataset is already normalized

    print(f'Opening dataset: {dataset_path}')
    f = h5py.File(dataset_path, "r+")
    demos = list(f["data"].keys())

    lengths=[]
    demos_minmax={}
    for demo_name in demos:
        demo=f['data'][demo_name]
        num_samples=demo.attrs['num_samples']
        lengths.append(num_samples)

        action=f['data'][demo_name]['actions']
        action=np.array(action) 
        demos_minmax[demo_name] = (np.min(action, axis=0), np.max(action, axis=0))


    lengths=np.array(lengths)

    print('Number of demos: ', len(demos))
    print('Max length: ', np.max(lengths))
    print('Min length: ', np.min(lengths))
    print('Mean length: ', np.mean(lengths))


    mins,maxs=[],[]
    for demo_name in demos_minmax.keys():
        min,max= demos_minmax[demo_name]
        mins.append(min)
        maxs.append(max)

    mins=np.min(mins, axis=0)
    maxs=np.max(maxs, axis=0) + 1e-8
    
    print('mins: ', mins)
    print('maxs: ', maxs)
    
    if np.all(mins==-1) and np.all(maxs==1):
        print('Dataset is already normalized')
        return

    for demo_name in demos:
        demo=f['data'][demo_name]  
        action=f['data'][demo_name]['actions']
        # convert action to [-1,1] using mins and maxs
        action= -1 + ( (action-mins)/(maxs-mins) )* 2.0

        del f["data"][demo_name]['actions']
        f["data"][demo_name].create_dataset('actions', data=action)


    # save stats to f as another key 
    f.create_dataset('mins', data=mins)
    f.create_dataset('maxs', data=maxs)

    f.close()

    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to dataset to normalize",
    )
    args = parser.parse_args()
    main(args.dataset)


# python action_normalize.py --dataset /home/ns/robosuite/collects/1705874644_525442/demo101_jan21_image_group.hdf5
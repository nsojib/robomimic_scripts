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
import glob
import argparse

def create_hdf5_filter_key(hdf5_path, demo_keys, key_name):
    """
    Creates a new hdf5 filter key in hdf5 file @hdf5_path with
    name @key_name that corresponds to the demonstrations
    @demo_keys. Filter keys are generally useful to create
    named subsets of the demonstrations in an hdf5, making it
    easy to train, test, or report statistics on a subset of
    the trajectories in a file.

    Returns the list of episode lengths that correspond to the filtering.

    Args:
        hdf5_path (str): path to hdf5 file
        demo_keys ([str]): list of demonstration keys which should
            correspond to this filter key. For example, ["demo_0", 
            "demo_1"].
        key_name (str): name of filter key to create

    Returns:
        ep_lengths ([int]): list of episode lengths that corresponds to
            each demonstration in the new filter key
    """
    f = h5py.File(hdf5_path, "a")  
    demos = sorted(list(f["data"].keys()))

    # collect episode lengths for the keys of interest
    ep_lengths = []
    for ep in demos:
        ep_data_grp = f["data/{}".format(ep)]
        if ep in demo_keys:
            ep_lengths.append(ep_data_grp.attrs["num_samples"])

    # store list of filtered keys under mask group
    k = "mask/{}".format(key_name)
    if k in f:
        del f[k]
    f[k] = np.array(demo_keys, dtype='S16')

    f.close()
    return ep_lengths


def main(dataset_path, group_videos):
    
    if group_videos is None:
        path=os.path.dirname(dataset_path)
        group_videos=path+"/videos/"
     

    if group_videos[-1]!='/':
        group_videos=group_videos+"/"

    
    
    print('dataset_path', dataset_path)
    print('group_videos', group_videos)
    groups=[f for f in glob.glob(group_videos+'*') if os.path.isdir(f) ]
    print('Total groups', len(groups)) 

    # return

    hdf5_file_name=dataset_path

    f = h5py.File(dataset_path, "r+")
    for group in groups:
        print('group', group)
        files=glob.glob(group+'/*.mp4')
        print('Total demos', len(files))
        group_demos=[os.path.basename(file).replace(".mp4", "") for file in files]

        print('group_demos', group_demos)
        
        # #save group
        group_demos = np.array(group_demos, dtype='S16') 


        # print(f'group_names after: {group_demos}')

        hdf5_path=hdf5_file_name 
        filter_keys=sorted([elem for elem in group_demos])
        filter_name=os.path.basename(group)
        filter_lengths = create_hdf5_filter_key(hdf5_path=hdf5_path, demo_keys=filter_keys, key_name=filter_name)

        print('\n\n: ', filter_name,  filter_lengths)
    f.close()


if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--dataset', type=str, help='path to input hdf5 dataset', required=True)
    args.add_argument('--group_videos', type=str, help='path to groups') 
    args = args.parse_args()
    main(args.dataset, args.group_videos)

# python create_groups.py --dataset /home/ns/robosuite/collects/1705874644_525442/demo101_jan21_image_group.hdf5 --group_videos /home/ns/robosuite/collects/1705874644_525442/videos
    
# python create_groups.py --dataset /home/ns/collect_robomimic_demos/PickPlaceCan_01_30_2024_01_19PM_sojib/demo_image.hdf5 
# python create_groups.py --dataset /home/ns/collect_robomimic_demos/PickPlaceCan_01_30_2024_01_45PM_sojib/demo_image.hdf5
# python create_groups.py --dataset /home/ns/collect_robomimic_demos/PickPlaceCan_01_30_2024_02_33PM_sojib/demo_image.hdf5

 #python create_groups.py --dataset /home/ns/collect_robomimic_demos/Lift_01_30_2024_05_03PM_sojib/demo_image.hdf5


 #python create_groups.py --dataset /home/ns/collect_robomimic_demos/Alyssa/PickPlaceCan_01_27_2024_04_35PM_AlyssaColandreo/demo_image.hdf5
    
    
    
    

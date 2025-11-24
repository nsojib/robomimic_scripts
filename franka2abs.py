### robomimic style data.

import h5py
import numpy as np
import shutil
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ========= config =========
src_path = "all_expert.hdf5"        # your original file
dst_path = "all_expert_abs.hdf5" # output file
compress = {"compression": "gzip", "compression_opts": 4}
# ==========================

src_path = Path(src_path)
dst_path = Path(dst_path)

# make a copy so we don't destroy the original
shutil.copy(src_path, dst_path)
print(f"Copied {src_path} -> {dst_path}")

with h5py.File(dst_path, "r+") as f:
    if "data" not in f:
        raise RuntimeError("Expected a 'data' group (robomimic-style).")

    data_grp = f["data"]

    for demo_name, demo_grp in data_grp.items():
        # typical robomimic layout: demo_grp["obs"], demo_grp["next_obs"], ...
        for obs_key in ["obs", "next_obs"]:
            if obs_key not in demo_grp:
                continue

            obs_grp = demo_grp[obs_key]
            print(f"Processing {demo_name}/{obs_key}")

            #rename image observations.
            rgb = obs_grp["agentview_rgb"][...]
            obs_grp.create_dataset("agentview_image", data=rgb, **compress) 

            eye = obs_grp["eye_in_hand_rgb"][...]
            obs_grp.create_dataset("robot0_eye_in_hand_image", data=eye, **compress) 

            # process joint positions
            j = obs_grp["joint_states"][...]
            obs_grp.create_dataset("robot0_joint_pos", data=j, **compress)

            # process end-effector pose
            ee_all = obs_grp["ee_states"][...]
            T_all = ee_all.reshape(-1, 4, 4, order='F')  # (N, 4, 4)
            R_all = T_all[:, :3, :3]                     # (N, 3, 3)
            pos_all = T_all[:, :3, 3]                    # (N, 3)

            quat_all = R.from_matrix(R_all).as_quat()    # (N, 4), [x,y,z,w]

            obs_grp.create_dataset("robot0_eef_pos", data=pos_all, **compress)
            obs_grp.create_dataset("robot0_eef_quat", data=quat_all, **compress)

            # process gripper states
            g = obs_grp["gripper_states"][...]
            g2 = np.concatenate([g, g], axis=-1)  # (T,2)
            obs_grp.create_dataset("robot0_gripper_qpos", data=g2, **compress)

print("Refactor done.")


import h5py
import numpy as np
import json

filepath = "/workspace/tudor_unitree_isaaclab/grasp_data2/episode_0000.hdf5"

with h5py.File(filepath, "r") as f:
    print(f.attrs["timestamp"])  # shows when this file was saved
    # --- Metadata ---
    print(json.loads(f.attrs["metadata"]))

    # --- Joint names (find index finger indices) ---
    joint_names = [n.decode() for n in f["joint_names"][:]]
    # Find index finger joints
    index_joints = [(i, n) for i, n in enumerate(joint_names) if "index" in n]
    print("Index finger joints:", index_joints)

    # --- Index finger positions over time ---
    joint_pos = f["timesteps/joint_pos"][:]    # (N, 53)
    timestamps = f["timesteps/timestamps"][:]  # (N,)
    for idx, name in index_joints:
        print(f"{name}: min={joint_pos[:, idx].min():.3f}  "
              f"max={joint_pos[:, idx].max():.3f}")

    # --- Tactile sensors ---
    print("\nTactile sensors:", list(f["tactile"].keys()))
    # Pick one index sensor (name depends on your config)
    for sensor_name in f["tactile"]:
        sg = f["tactile"][sensor_name]
        print(f"\n{sensor_name}:")
        for key in sg:
            print(f"  {key}: shape={sg[key].shape}, "
                f"min={sg[key][:].min():.4f}, max={sg[key][:].max():.4f}")
        # if "little" in sensor_name:
        #     sg = f["tactile"][sensor_name]
        #     print(f"\n{sensor_name}:")
        #     for key in sg:
        #         print(f"  {key}: shape={sg[key].shape}, "
        #               f"min={sg[key][:].min():.4f}, max={sg[key][:].max():.4f}")

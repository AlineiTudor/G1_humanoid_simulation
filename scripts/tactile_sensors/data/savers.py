"""
Data Savers - HDF5 episode storage
"""

import json
import numpy as np
import h5py
from typing import Dict, Any, List
from datetime import datetime


def save_episode_hdf5(episode_data: Dict[str, Any],
                      filepath: str,
                      metadata: Dict[str, Any] = None,
                      joint_names: List[str] = None):
    """Save a single grasp episode to HDF5 format.

    Args:
        epsiode_data: Dict from GraspEpsiodeCollector.get_episode_data().
        filepath: Output .hdf5 path.
        metadata: Dict of experiment metadata (config, etc.).
        joint_names: List of joint names for reference.
    
    """
    with h5py.File(filepath, "w") as f:
        # Metadata as JSON attribute
        if metadata:
            f.attrs["metadata"] = json.dumps(metadata, default = str)
        f.attrs["timestamp"] = datetime.now().isoformat()

        # Joint names
        if joint_names:
            dt = h5py.string_dtype()
            f.create_dataset("joint_names", data = joint_names, dtype=dt)
        
        # Timestamp arrays
        ts = f.create_group("timesteps")
        for key in ["timestamps", "joint_pos", "joint_vel", "joint_torque",
                     "actions", "object_pos", "object_rot", "grasp_phase"]:
            arr = episode_data.get(key)
            if arr is not None and len(arr) > 0:
                ts.create_dataset(key, data=arr, compression = "gzip", compression_opts=4)
        
        # Tactile data - per sensor
        tactile = episode_data.get("tactile", {})
        if tactile:
            tac_group = f.create_group("tactile",{})
            for sensor_name, sensor_data in tactile.items():
                if not sensor_data:
                    continue
                
                sg = tac_group.create_group(sensor_name)
                for data_key, data_array in sensor_data.items():
                    if data_array is not None and len(data_array) > 0:
                        sg.create_dataset(data_key, data = data_array, compression = "gzip", compression_opts = 4)
        
        # Camera RGB
        camera_rgb = episode_data.get("camera_rgb")
        if camera_rgb is not None and len(camera_rgb) > 0:
            f.create_dataset("camera/rgb", data = camera_rgb, compression="gzip", compression_opts = 4)

def save_metadata(config_dict: Dict[str, Any], filepath: str):
    """Save experiment configuration as JSON."""
    with open(filepath, "w") as f:
        json.dump(config_dict, f, indent=2, default=2)
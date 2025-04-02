import json
import torch
from easydict import EasyDict

from libero.lifelong.metric import raw_obs_to_tensor_obs
from libero.lifelong.models.bc_transformer_policy import BCTransformerPolicy

import robomimic.utils.obs_utils as ObsUtils

MODALITY_CONFIG = EasyDict({
    "data": {
        "obs":{
            "modality":{
                "rgb":[
                "agentview_rgb",
                "eye_in_hand_rgb"
                ],
                "depth":[
                
                ],
                "low_dim":[
                "gripper_states",
                "joint_states"
                ]
            }
        },
        "obs_key_mapping":{
            "agentview_rgb":"agentview_image",
            "eye_in_hand_rgb":"robot0_eye_in_hand_image",
            "gripper_states":"robot0_gripper_qpos",
            "joint_states":"robot0_joint_pos"
        },
    }
})
ObsUtils.initialize_obs_utils_with_obs_specs({"obs": MODALITY_CONFIG.data.obs.modality})

def obs_to_tensor(obs_tensor_dict):
    obs_tensor_dict = raw_obs_to_tensor_obs(obs_tensor_dict, torch.randn(1, 16), MODALITY_CONFIG)['obs']
    # Flatten each observation type
    img1 = obs_tensor_dict["agentview_rgb"].flatten(start_dim=1)
    img2 = obs_tensor_dict["eye_in_hand_rgb"].flatten(start_dim=1)
    gripper = obs_tensor_dict["gripper_states"].flatten(start_dim=1)
    joints = obs_tensor_dict["joint_states"].flatten(start_dim=1)

    # Concatenate everything along the last dimension
    return torch.cat([img1, img2, gripper, joints], dim=1)  # (batch, total_features)


def make_policy(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    config = EasyDict(config)
    return BCTransformerPolicy(config, config['shape_meta'])


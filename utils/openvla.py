# This code is modified from: https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py


"""
run_libero_eval.py

Runs a model in a LIBERO simulation environment.

Usage:
    # OpenVLA:
    # IMPORTANT: Set `center_crop=True` if model is fine-tuned with augmentations
    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint <CHECKPOINT_PATH> \
        --task_suite_name [ libero_spatial | libero_object | libero_goal | libero_10 | libero_90 ] \
        --center_crop [ True | False ] \
        --run_id_note <OPTIONAL TAG TO INSERT INTO RUN ID FOR LOGGING> \
        --use_wandb [ True | False ] \
        --wandb_project <PROJECT> \
        --wandb_entity <ENTITY>
"""

import os
from pathlib import Path
from typing import Union

import numpy as np
from libero.libero import benchmark

# Append current directory so that interpreter can find experiments.robot
# sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

from easydict import EasyDict

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.venv import DummyVectorEnv

from libero.lifelong.utils import get_task_embs


class OpenVLAConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path
    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"          # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    # num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    # num_trials_per_task: int = 50                    # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    # run_id_note: Optional[str] = None                # Extra note to add in run ID for logging
    # local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    # use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    # wandb_project: str = "YOUR_WANDB_PROJECT"        # Name of W&B project to log to (use default!)
    # wandb_entity: str = "YOUR_WANDB_ENTITY"          # Name of entity to log under

    seed: int = 7                                    # Random Seed (for reproducibility)

    # fmt: on


EMBED_CONFIG = EasyDict({
    "task_embedding_format": "bert",
    "data": {
        "max_word_len": 25,
    },
    "policy": {
        "language_encoder": {
            "network_kwargs": {
                "input_size": 768,
            }
        }
    }
})


def get_libero_env(task, horizon=600, resolution=256):
    """Initializes and returns the LIBERO environment."""
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution, 
        "camera_widths": resolution,
        "horizon": horizon
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env

def make_libero_envs(
        num_envs=2, 
        task_suite_name='libero_object', 
        task_id=0,
        horizon=600,
        camera_dim=128,
    ):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_description = task.language
    task_emb = get_task_embs(EMBED_CONFIG, [task_description]).squeeze()

    # Get default LIBERO initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # Initialize LIBERO environment and task description
    
    envs = DummyVectorEnv(
        [lambda: get_libero_env(task, horizon=horizon, resolution=camera_dim) for _ in range(num_envs)]
    )
    envs.reset()

    return envs, initial_states, task, task_emb

def get_openvla_model(cfg: OpenVLAConfig):
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    if "image_aug" in cfg.pretrained_checkpoint:
        assert cfg.center_crop, "Expecting `center_crop==True` because model was trained with image augmentations!"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "Cannot use both 8-bit and 4-bit quantization!"

    # Set random seed
    set_seed_everywhere(cfg.seed)

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = cfg.task_suite_name

    # Load model
    model = get_model(cfg)

    # [OpenVLA] Check that the model contains the action un-normalization key
    if cfg.model_family == "openvla":
        # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
        # with the suffix "_no_noops" in the dataset name)
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"Action un-norm key {cfg.unnorm_key} not found in VLA `norm_stats`!"

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    
    return processor, model

def get_openvla_action(cfg, model, obs, task_description, processor):
    resize_size = get_image_resize_size(cfg)
    img = get_libero_image(obs, resize_size)

    # Prepare observations dict
    # Note: OpenVLA does not take proprio state as input
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    # Query model to get action
    action = get_action(
        cfg,
        model,
        observation,
        task_description,
        processor=processor,
    )

    action = normalize_gripper_action(action, binarize=True)

    # [OpenVLA] The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    if cfg.model_family == "openvla":
        action = invert_gripper_action(action)

    return action


def get_dummy_action(cfg):
    return get_libero_dummy_action(cfg.model_family)

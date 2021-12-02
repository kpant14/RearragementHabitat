# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

from numpy import random
from .rearrange_env import RearrangementRLEnv
from env.rearrange_dataset import RearrangementDatasetV0
import numpy as np
import torch
from habitat.config.default import get_config as cfg_env
from habitat.core.env import Env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from .exploration_env import Exploration_Env
from .habitat_lab.habitat.core.vector_env import VectorEnv
from .habitat_lab.habitat_baselines.config.default import get_config as cfg_baseline


def make_env_fn(args, config_env, config_baseline, rank):
    #dataset = PointNavDatasetV1(config_env.DATASET)
    dataset = RearrangementDatasetV0(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.SIMULATOR.SCENE))
    config_env.freeze()
    # env = Exploration_Env(args=args, rank=rank,
    #                      config_env=config_env, config_baseline=config_baseline, dataset=dataset
    #                      )
    env = RearrangementRLEnv(args=args, rank=rank,config_env=config_env, config_baseline=config_baseline, dataset=dataset)
    env.seed(rank)
    return env


def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_paths=
                           ["env/habitat/habitat_lab/configs/" + args.task_config])                       
    basic_config.defrost()
    basic_config.DATASET.SPLIT = args.split
    basic_config.DATASET.DATA_PATH = (
    "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
    )
    basic_config.DATASET.TYPE = "RearrangementDataset-v0"
    basic_config.freeze()
    scenes  = RearrangementDatasetV0.get_scenes_to_load(basic_config.DATASET)
    #scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.DATASET)
    random.shuffle(scenes)
    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))

    for i in range(args.num_processes):
        config_env = cfg_env(config_paths=
                             ["env/habitat/habitat_lab/configs/" + args.task_config])
        config_env.defrost()
        config_env.DATASET.DATA_PATH = (
        "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
        )
        config_env.DATASET.TYPE = "RearrangementDataset-v0"
        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 50
        config_env.SIMULATOR.TYPE = "RearrangementSim-v0"
        config_env.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
        config_env.SIMULATOR.GRAB_DISTANCE = 2.0
        config_env.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
        config_env.TASK.TYPE = "RearrangementTask-v0"
        config_env.TASK.SUCCESS_DISTANCE = 1.0
        config_env.TASK.SENSORS = [
            "GRIPPED_OBJECT_SENSOR",
            "OBJECT_POSITION",
            "OBJECT_GOAL",
        ]
        config_env.TASK.GOAL_SENSOR_UUID = "object_goal"
        config_env.TASK.MEASUREMENTS = [
            "OBJECT_TO_GOAL_DISTANCE",
            "AGENT_TO_OBJECT_DISTANCE",
        ]
        config_env.TASK.POSSIBLE_ACTIONS = ["STOP","MOVE_FORWARD","TURN_LEFT","TURN_RIGHT","GRAB_RELEASE"]
        if len(scenes) > 0:
            config_env.DATASET.CONTENT_SCENES = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = gpu_id

        agent_sensors = []
        agent_sensors.append("RGB_SENSOR")
        agent_sensors.append("DEPTH_SENSOR")

        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors

        config_env.ENVIRONMENT.MAX_EPISODE_STEPS = args.max_episode_length
        config_env.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False

        config_env.SIMULATOR.RGB_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.RGB_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.DEPTH_SENSOR.WIDTH = args.env_frame_width
        config_env.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.env_frame_height
        config_env.SIMULATOR.DEPTH_SENSOR.HFOV = args.hfov
        config_env.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]

        config_env.SIMULATOR.TURN_ANGLE = 10
        config_env.DATASET.SPLIT = args.split
        config_env.freeze()
        env_configs.append(config_env)
        config_baseline = cfg_baseline()
        #config_baseline = config_env
        baseline_configs.append(config_baseline)
        
        args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(args_list, env_configs, baseline_configs,
                    range(args.num_processes))
            )
        ),
    )
    return envs

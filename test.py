import random

import numpy as np
import torch
from env.rearrange_baseline_policy import RearrangementTrainer

import habitat
from habitat import Config
from habitat_baselines.config.default import get_config as get_baseline_config
from arguments import get_args

config = habitat.get_config("configs/tasks/pointnav.yaml")
config.defrost()
config.ENVIRONMENT.MAX_EPISODE_STEPS = 50
config.SIMULATOR.TYPE = "RearrangementSim-v0"
config.SIMULATOR.ACTION_SPACE_CONFIG = "RearrangementActions-v0"
config.SIMULATOR.GRAB_DISTANCE = 2.0
config.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = True
config.TASK.TYPE = "RearrangementTask-v0"
config.TASK.SUCCESS_DISTANCE = 1.0
config.TASK.SENSORS = [
    "GRIPPED_OBJECT_SENSOR",
    "OBJECT_POSITION",
    "OBJECT_GOAL",
]
config.TASK.GOAL_SENSOR_UUID = "object_goal"
config.TASK.MEASUREMENTS = [
    "OBJECT_TO_GOAL_DISTANCE",
    "AGENT_TO_OBJECT_DISTANCE",
]
config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "GRAB_RELEASE"]
config.DATASET.TYPE = "RearrangementDataset-v0"
config.DATASET.SPLIT = "train"
config.DATASET.DATA_PATH = (
    "data/datasets/rearrangement/coda/v1/{split}/{split}.json.gz"
)
config.freeze()

baseline_config = get_baseline_config(
    "env/habitat/habitat_lab/habitat_baselines/config/pointnav/ppo_pointnav.yaml"
)
baseline_config.defrost()

baseline_config.TASK_CONFIG = config
baseline_config.TRAINER_NAME = "ddppo"
baseline_config.ENV_NAME = "RearrangementRLEnv"
baseline_config.SIMULATOR_GPU_ID = 0
baseline_config.TORCH_GPU_ID = 0
baseline_config.VIDEO_OPTION = ["disk"]
baseline_config.TENSORBOARD_DIR = "data/tb"
baseline_config.VIDEO_DIR = "data/videos"
baseline_config.NUM_ENVIRONMENTS = 2
baseline_config.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
baseline_config.CHECKPOINT_FOLDER = "data/checkpoints"
baseline_config.TOTAL_NUM_STEPS = -1.0


baseline_config.NUM_UPDATES = 400  # @param {type:"number"}

baseline_config.LOG_INTERVAL = 10
baseline_config.NUM_CHECKPOINTS = 5
baseline_config.LOG_FILE = "data/checkpoints/train.log"
baseline_config.EVAL.SPLIT = "train"
baseline_config.RL.SUCCESS_REWARD = 2.5  # @param {type:"number"}
baseline_config.RL.SUCCESS_MEASURE = "object_to_goal_distance"
baseline_config.RL.REWARD_MEASURE = "object_to_goal_distance"
baseline_config.RL.GRIPPED_SUCCESS_REWARD = 2.5  # @param {type:"number"}

baseline_config.freeze()
random.seed(baseline_config.TASK_CONFIG.SEED)
np.random.seed(baseline_config.TASK_CONFIG.SEED)
torch.manual_seed(baseline_config.TASK_CONFIG.SEED)

if __name__ == "__main__":
    args = get_args()
    trainer = RearrangementTrainer(baseline_config,args)
    trainer.train()
    trainer.eval()

    # if make_video:
    #     video_file = os.listdir("data/videos")[0]
    #     vut.display_video(os.path.join("data/videos", video_file))
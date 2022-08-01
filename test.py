import random

import numpy as np
import torch
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
from habitat_baselines.utils.env_utils import construct_envs
from rearrange_baseline_policy import RearrangementTrainer

import habitat
from habitat import Config
from habitat_baselines.config.default import get_config as get_baseline_config
from arguments import get_args
from env.rearrange_dataset import RearrangementDatasetV0

import numpy as np
from typing import Optional, Dict, List
import json
from habitat.config.default import CN, Config
from habitat import Config, Dataset
import habitat

config = habitat.get_config("configs/tasks/pointnav.yaml")
config.defrost()
config.ENVIRONMENT.MAX_EPISODE_STEPS = 500
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
    "data/datasets/rearrangement/gibson/v1/{split}/{split}.json.gz"
)
config.freeze()

baseline_config = get_baseline_config(
    "env/habitat/habitat-lab/habitat_baselines/config/pointnav/ppo_pointnav.yaml"
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
baseline_config.NUM_ENVIRONMENTS = 1
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
baseline_config.TASK_CONFIG = config
baseline_config.freeze()
random.seed(baseline_config.TASK_CONFIG.SEED)
np.random.seed(baseline_config.TASK_CONFIG.SEED)
torch.manual_seed(baseline_config.TASK_CONFIG.SEED)


class RearrangementRLEnv(habitat.RLEnv):
    #def __init__(self, args, rank, config_env, config_baseline, dataset):
    def __init__(
        self, config: Config, dataset: Optional[Dataset] = None
    ) -> None:
        self._prev_measure = {
            "agent_to_object_distance": None,
            "object_to_goal_distance": None,
            "gripped_object_id": -1,
            "gripped_object_count": 0,
        }
        #super().__init__(args, rank,config_env, config_baseline, dataset)
       
        self.num_actions = 3
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._previous_action = None
        #self.args = args
        self._success_distance = config.TASK_CONFIG.TASK.SUCCESS_DISTANCE
        task_config = config.TASK_CONFIG
        super().__init__(task_config, dataset)
       
    def reset(self):
        self._previous_action = None
        obs = super().reset()
        #state, self.info = super().reset()
        self._prev_measure.update(self.habitat_env.get_metrics())
        self._prev_measure["gripped_object_id"] = -1
        self._prev_measure["gripped_object_count"] = 0
        # self.info["agent_to_object_distance"] = self._prev_measure["agent_to_object_distance"]
        # self.info["object_to_goal_distance"] = self._prev_measure["object_to_goal_distance"]
        # self.info["gripped_object_id"] = self._prev_measure["gripped_object_id"]
        # self.info["gripped_object_count"] = self._prev_measure["gripped_object_count"]
        # return state, self.info
        obs['object_position'] = obs['object_position'][0]
        obs['object_goal'] = obs['object_goal'][0] 
        #print(observations['object_position'])
        return obs

    def step(self, action):
        self._previous_action = action
        obs, rew, done, info = super().step(action)
        # self.info["agent_to_object_distance"] = self._prev_measure["agent_to_object_distance"]
        # self.info["object_to_goal_distance"] = self._prev_measure["object_to_goal_distance"]
        # self.info["gripped_object_id"] = self._prev_measure["gripped_object_id"]
        # self.info["gripped_object_count"] = self._prev_measure["gripped_object_count"]
        obs['object_position'] = obs['object_position'][0]
        obs['object_goal'] = obs['object_goal'][0] 
        return obs, rew, done, info     

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        #reward = self._rl_config.SLACK_REWARD
        reward = 0
        gripped_success_reward = 0.0
        episode_success_reward = 0.0
        agent_to_object_dist_reward = 0.0
        object_to_goal_dist_reward = 0.0

        action_name = self._env.task.get_action_name(
            self._previous_action
        )
        # If object grabbed, add a success reward
        # The reward gets awarded only once for an object.
        if (
            action_name == "GRAB_RELEASE"
            and observations["gripped_object_id"] >= 0
        ):
            obj_id = observations["gripped_object_id"]
            self._prev_measure["gripped_object_count"] += 1

            gripped_success_reward = (
                self._rl_config.GRIPPED_SUCCESS_REWARD
                if self._prev_measure["gripped_object_count"] == 1
                else 0.0
            )
        # add a penalty everytime grab/action is called and doesn't do anything
        elif action_name == "GRAB_RELEASE":
            gripped_success_reward += -0.1

        self._prev_measure["gripped_object_id"] = observations[
            "gripped_object_id"
        ]

        # If the action is not a grab/release action, and the agent
        # has not picked up an object, then give reward based on agent to
        # object distance.
        if (
            action_name != "GRAB_RELEASE"
            and self._prev_measure["gripped_object_id"] == -1
        ):
            agent_to_object_dist_reward = self.get_agent_to_object_dist_reward(
                observations
            )

        # If the action is not a grab/release action, and the agent
        # has picked up an object, then give reward based on object to
        # to goal distance.
        if (
            action_name != "GRAB_RELEASE"
            and self._prev_measure["gripped_object_id"] != -1
        ):
            object_to_goal_dist_reward = self.get_object_to_goal_dist_reward()

        if (
            self._episode_success(observations)
            and self._prev_measure["gripped_object_id"] == -1
            and action_name == "STOP"
        ):
            episode_success_reward = self._rl_config.SUCCESS_REWARD
        reward += (
            agent_to_object_dist_reward
            + object_to_goal_dist_reward
            + gripped_success_reward
            + episode_success_reward
        )
        return reward

    def get_agent_to_object_dist_reward(self, observations):
        """
        Encourage the agent to move towards the closest object which is not already in place.
        """
        curr_metric = self._env.get_metrics()["agent_to_object_distance"]
        prev_metric = self._prev_measure["agent_to_object_distance"]
        dist_reward = prev_metric - curr_metric
        self._prev_measure["agent_to_object_distance"] = curr_metric
        return np.sum(dist_reward[0])# only for first object

    def get_object_to_goal_dist_reward(self):
        curr_metric = self._env.get_metrics()["object_to_goal_distance"]
        prev_metric = self._prev_measure["object_to_goal_distance"]
        dist_reward = prev_metric - curr_metric
        self._prev_measure["object_to_goal_distance"] = curr_metric
        return np.sum(dist_reward[0])# only for first object

    def _episode_success(self, observations):
        r"""Returns True if object is within distance threshold of the goal."""
        dist = self._env.get_metrics()["object_to_goal_distance"]
        if (
            abs(dist).all() > self._success_distance
            or observations["gripped_object_id"] != -1
        ):
            return False
        return True

    def _gripped_success(self, observations):
        if (
            observations["gripped_object_id"] >= 0
            and observations["gripped_object_id"]
            != self._prev_measure["gripped_object_id"]
        ):
            return True

        return False

    def get_done(self, observations):
        done = False
        action_name = self._env.task.get_action_name(
            self._previous_action
        )
        if self._env.episode_over or (
            self._episode_success(observations)
            and self._prev_measure["gripped_object_id"] == -1
            and action_name == "STOP"
        ):
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        return info

            
if __name__ == "__main__":
    dataset = RearrangementDatasetV0(config.DATASET)
    env = RearrangementRLEnv(baseline_config, dataset)
    env.reset()
    env.step(1)

    envs = construct_envs(
        baseline_config,
        habitat.RLEnv,
        workers_ignore_signals=is_slurm_batch_job(),
    )
    envs.reset()
    envs.step([1])
    # trainer = RearrangementTrainer(baseline_config,None)
    # trainer.train()
    # trainer.eval()

    # if make_video:
    #     video_file = os.listdir("data/videos")[0]
    #     vut.display_video(os.path.join("data/videos", video_file))





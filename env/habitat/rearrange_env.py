from env.habitat.exploration_env import Exploration_Env
from typing import Optional, Dict, List
import json
from habitat.config.default import CN, Config
from habitat import Config, Dataset
from habitat_baselines.common.environments import NavRLEnv
import gym

class RearrangementRLEnv(Exploration_Env):
    def __init__(self, args, rank, config_env, config_baseline, dataset):
        self._prev_measure = {
            "agent_to_object_distance": None,
            "object_to_goal_distance": None,
            "gripped_object_id": -1,
            "gripped_object_count": 0,
        }
        super().__init__(args, rank,config_env, config_baseline, dataset)
        self.num_actions = 3
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.observation_space = gym.spaces.Box(0, 255,
                                                (3, args.frame_height,
                                                    args.frame_width),
                                                dtype='uint8')
        #self._rl_config = config_env.RL
        #self._core_env_config = config_env.TASK_CONFIG
        self._previous_action = None
        self.args = args
        self._success_distance = config_env.TASK.SUCCESS_DISTANCE
       
    def reset(self):
        self._previous_action = None
        #observations = super().reset()
        state, self.info = super().reset()
        self._prev_measure.update(self.habitat_env.get_metrics())
        self._prev_measure["gripped_object_id"] = -1
        self._prev_measure["gripped_object_count"] = 0
        self.info["agent_to_object_distance"] = self._prev_measure["agent_to_object_distance"]
        self.info["object_to_goal_distance"] = self._prev_measure["object_to_goal_distance"]
        self.info["gripped_object_id"] = self._prev_measure["gripped_object_id"]
        self.info["gripped_object_count"] = self._prev_measure["gripped_object_count"]
        return state, self.info

    def step(self, action):
        self._previous_action = action
        state, rew, done, self.info = super().step(action)
        self.info["agent_to_object_distance"] = self._prev_measure["agent_to_object_distance"]
        self.info["object_to_goal_distance"] = self._prev_measure["object_to_goal_distance"]
        self.info["gripped_object_id"] = self._prev_measure["gripped_object_id"]
        self.info["gripped_object_count"] = self._prev_measure["gripped_object_count"]
        return state, rew, done, self.info     

    def get_reward_range(self):
        return (
            #self._rl_config.SLACK_REWARD - 1.0,
            #self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        #reward = self._rl_config.SLACK_REWARD
        reward = 0
        gripped_success_reward = 0.0
        episode_success_reward = 0.0
        agent_to_object_dist_reward = 0.0
        object_to_goal_dist_reward = 0.0

        # action_name = self._env.task.get_action_name(
        #     self._previous_action["action"]
        # )
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
        return dist_reward

    def get_object_to_goal_dist_reward(self):
        curr_metric = self._env.get_metrics()["object_to_goal_distance"]
        prev_metric = self._prev_measure["object_to_goal_distance"]
        dist_reward = prev_metric - curr_metric
        self._prev_measure["object_to_goal_distance"] = curr_metric
        return dist_reward

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
        # action_name = self._env.task.get_action_name(
        #     self._previous_action["action"]
        # )
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

        
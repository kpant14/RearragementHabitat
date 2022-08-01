import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import torch
from env.habitat.rearrange_env import RearrangementRLEnv

from habitat import Config, logger
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.ddp_utils import is_slurm_batch_job
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import PPO
from habitat_baselines.rl.ppo.policy import Net, Policy
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.env_utils import construct_envs, make_env_fn
from env.rearrange_task import ObjectGoal, ObjectPosition
#from env import construct_envs

class RearrangementBaselinePolicy(Policy):
    def __init__(self, observation_space, action_space, hidden_size=512):
        super().__init__(
            RearrangementBaselineNet(
                observation_space=observation_space, hidden_size=hidden_size
            ),
            action_space.n,
        )

    def from_config(cls, config, envs):
        pass


class RearrangementBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size):
        super().__init__()

        self._n_input_goal = observation_space.spaces[
            ObjectGoal.cls_uuid
        ].shape[0]

        self._hidden_size = hidden_size

        self.state_encoder = build_rnn_state_encoder(
            2 * self._n_input_goal, self._hidden_size
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        object_goal_encoding = observations[ObjectGoal.cls_uuid]
        object_pos_encoding = observations[ObjectPosition.cls_uuid]

        x = [object_goal_encoding, object_pos_encoding]

        x = torch.cat(x, dim=1)
        x, rnn_hidden_states = self.state_encoder(x, rnn_hidden_states, masks)

        return x, rnn_hidden_states


@baseline_registry.register_trainer(name="ppo-rearrangement")
class RearrangementTrainer(PPOTrainer):
    supported_tasks = ["RearrangementTask-v0"]

    def __init__(self, config=None, args=None):
        super().__init__(config)
        self.args = args
    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = RearrangementBaselinePolicy(
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
        )
        self.actor_critic.to(self.device)
        
        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            RearrangementRLEnv,
            workers_ignore_signals=is_slurm_batch_job(),
        )
        self.envs.reset()

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("This trainer does not support distributed")
        self._init_train()

        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda _: 1 - self.percent_done(),
        )
        ppo_cfg = self.config.RL.PPO

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            while not self.is_done():

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                count_steps_delta = 0
                for _step in range(ppo_cfg.num_steps):
                    count_steps_delta += self._collect_rollout_step()

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                losses = self._coalesce_post_step(
                    dict(value_loss=value_loss, action_loss=action_loss),
                    count_steps_delta,
                )
                self.num_updates_done += 1

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in self.window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "reward",
                    deltas["reward"] / deltas["count"],
                    self.num_steps_done,
                )

                # Check to see if there are any metrics
                # that haven't been logged yet

                for k, v in deltas.items():
                    if k not in {"reward", "count"}:
                        writer.add_scalar(
                            "metric/" + k,
                            v / deltas["count"],
                            self.num_steps_done,
                        )

                losses = [value_loss, action_loss]
                for l, k in zip(losses, ["value, policy"]):
                    writer.add_scalar("losses/" + k, l, self.num_steps_done)

                # log stats
                if self.num_updates_done % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            self.num_updates_done,
                            self.num_steps_done / (time.time() - self.t_start),
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            self.num_updates_done,
                            self.env_time,
                            self.pth_time,
                            self.num_steps_done,
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(self.window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(step=self.num_steps_done),
                    )
                    count_checkpoints += 1

            self.envs.close()

    def eval(self) -> None:
        r"""Evaluates the current model
        Returns:
            None
        """

        config = self.config.clone()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.NUM_ENVIRONMENTS = 1
            config.freeze()

        logger.info(f"env config: {config}")
        with construct_envs(self.args) as envs:
            observations = envs.reset()
            batch = batch_obs(observations, device=self.device)

            current_episode_reward = torch.zeros(
                envs.num_envs, 1, device=self.device
            )
            ppo_cfg = self.config.RL.PPO
            test_recurrent_hidden_states = torch.zeros(
                config.NUM_ENVIRONMENTS,
                self.actor_critic.net.num_recurrent_layers,
                ppo_cfg.hidden_size,
                device=self.device,
            )
            prev_actions = torch.zeros(
                config.NUM_ENVIRONMENTS,
                1,
                device=self.device,
                dtype=torch.long,
            )
            not_done_masks = torch.zeros(
                config.NUM_ENVIRONMENTS,
                1,
                device=self.device,
                dtype=torch.bool,
            )

            rgb_frames = [
                [] for _ in range(self.config.NUM_ENVIRONMENTS)
            ]  # type: List[List[np.ndarray]]

            if len(config.VIDEO_OPTION) > 0:
                os.makedirs(config.VIDEO_DIR, exist_ok=True)

            self.actor_critic.eval()

            for _i in range(config.TASK_CONFIG.ENVIRONMENT.MAX_EPISODE_STEPS):
                current_episodes = envs.current_episodes()

                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

                    prev_actions.copy_(actions)

                outputs = envs.step([a[0].item() for a in actions])

                observations, rewards, dones, infos = [
                    list(x) for x in zip(*outputs)
                ]
                batch = batch_obs(observations, device=self.device)

                not_done_masks = torch.tensor(
                    [[not done] for done in dones],
                    dtype=torch.bool,
                    device="cpu",
                )

                rewards = torch.tensor(
                    rewards, dtype=torch.float, device=self.device
                ).unsqueeze(1)

                current_episode_reward += rewards

                # episode ended
                if not not_done_masks[0].item():
                    generate_video(
                        video_option=self.config.VIDEO_OPTION,
                        video_dir=self.config.VIDEO_DIR,
                        images=rgb_frames[0],
                        episode_id=current_episodes[0].episode_id,
                        checkpoint_idx=0,
                        metrics=self._extract_scalars_from_info(infos[0]),
                        tb_writer=None,
                    )

                    print("Evaluation Finished.")
                    print("Success: {}".format(infos[0]["episode_success"]))
                    print(
                        "Reward: {}".format(current_episode_reward[0].item())
                    )
                    print(
                        "Distance To Goal: {}".format(
                            infos[0]["object_to_goal_distance"]
                        )
                    )

                    return

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(observations[0], infos[0])
                    rgb_frames[0].append(frame)

                not_done_masks = not_done_masks.to(device=self.device)
# @title Implement new sensors and measurements
# @markdown After defining the dataset, action space and simulator functions for the rearrangement task, we are one step closer to training agents to solve this task.

# @markdown Here we define inputs to the policy and other measurements required to design reward functions.

# @markdown **Sensors**: These define various part of the simulator state that's visible to the agent. For simplicity, we'll assume that agent knows the object's current position, object's final goal position relative to the agent's current position.
# @markdown - Object's current position will be made given by the `ObjectPosition` sensor
# @markdown - Object's goal position will be available through the `ObjectGoal` sensor.
# @markdown - Finally, we will also use `GrippedObject` sensor to tell the agent if it's holding any object or not.

# @markdown **Measures**: These define various metrics about the task which can be used to measure task progress and define rewards. Note that measurements are *privileged* information not accessible to the agent as part of the observation space. We will need the following measurements:
# @markdown - `AgentToObjectDistance` which measure the euclidean distance between the agent and the object.
# @markdown - `ObjectToGoalDistance` which measures the euclidean distance between the object and the goal.

from gym import spaces

import habitat_sim
import numpy as np
from habitat.config.default import _C,CN, Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import Measure
from habitat.core.simulator import Observations, Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import PointGoalSensor
from habitat.core.registry import registry
from typing import Any, List, Dict, Type
from env.rearrange_sim import RearrangementSim
from habitat.tasks.nav.nav import NavigationTask, merge_sim_episode_config

@registry.register_sensor
class GrippedObjectSensor(Sensor):
    cls_uuid = "gripped_object_id"

    def __init__(
        self, *args: Any, sim: RearrangementSim, config: Config, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any):

        return spaces.Discrete(len(self._sim.get_existing_object_ids()))

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: Episode,
        *args: Any,
        **kwargs: Any,
    ):
        obj_id = self._sim.sim_object_to_objid_mapping.get(
            self._sim.gripped_object_id, -1
        )
        return obj_id


@registry.register_sensor
class ObjectPosition(PointGoalSensor):
    cls_uuid: str = "object_position"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        object_count = len(self._sim.get_existing_object_ids())
        pointgoal = np.zeros((object_count,2))
        for i,object_id in enumerate(self._sim.get_existing_object_ids()):
            object_position = self._sim.get_translation(object_id)
            pointgoal[i] = self._compute_pointgoal(agent_position, rotation_world_agent, object_position)   
        # object_position = self._sim.get_translation(object_id)
        # pointgoal = self._compute_pointgoal(
        #     agent_position, rotation_world_agent, object_position
        # )
        return pointgoal


@registry.register_sensor
class ObjectGoal(PointGoalSensor):
    cls_uuid: str = "object_goal"

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)

        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        object_count = len(self._sim.get_existing_object_ids())
        pointgoal = np.zeros((object_count,2))
        for i in range(object_count):
            goal_position = np.array(episode.goals[i].position, dtype=np.float32)
            pointgoal[i] = self._compute_pointgoal(agent_position, rotation_world_agent, goal_position)   

        # goal_position = np.array(episode.goals[0].position, dtype=np.float32)

        # point_goal = self._compute_pointgoal(
        #     agent_position, rotation_world_agent, goal_position
        # )
        return pointgoal


@registry.register_measure
class ObjectToGoalDistance(Measure):
    """The measure calculates distance of object towards the goal."""

    cls_uuid: str = "object_to_goal_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return ObjectToGoalDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _geo_dist(self, src_pos, goal_pos: np.array) -> float:
        return self._sim.geodesic_distance(src_pos, [goal_pos])

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_count = len(self._sim.get_existing_object_ids())
        metric = np.zeros((object_count,1))
        for i,sim_obj_id in enumerate(self._sim.get_existing_object_ids()):
            previous_position = np.array(self._sim.get_translation(sim_obj_id)).tolist()
            goal_position = episode.goals[i].position
            metric[i] = self._euclidean_distance(previous_position, goal_position)
        self._metric = metric  
        # sim_obj_id = self._sim.get_existing_object_ids()[0]

        # previous_position = np.array(
        #     self._sim.get_translation(sim_obj_id)
        # ).tolist()
        # #goal_position = episode.goals.position
        # goal_position = episode.goals[0].position
        # self._metric = self._euclidean_distance(
        #     previous_position, goal_position
        # )


@registry.register_measure
class AgentToObjectDistance(Measure):
    """The measure calculates the distance of objects from the agent"""

    cls_uuid: str = "agent_to_object_distance"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    @staticmethod
    def _get_uuid(*args: Any, **kwargs: Any):
        return AgentToObjectDistance.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self.update_metric(*args, episode=episode, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, *args: Any, **kwargs: Any):
        object_count = len(self._sim.get_existing_object_ids())
        metric = np.zeros((object_count,1))
        for i,sim_obj_id in enumerate(self._sim.get_existing_object_ids()):
            previous_position = np.array(self._sim.get_translation(sim_obj_id)).tolist()
            agent_state = self._sim.get_agent_state()
            agent_position = agent_state.position
            metric[i] = self._euclidean_distance(previous_position, agent_position)
        self._metric = metric   
        # sim_obj_id = self._sim.get_existing_object_ids()[0]
        # previous_position = np.array(
        #     self._sim.get_translation(sim_obj_id)
        # ).tolist()

        # agent_state = self._sim.get_agent_state()
        # agent_position = agent_state.position

        # self._metric = self._euclidean_distance(
        #     previous_position, agent_position
        # )


# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK GRIPPED OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GRIPPED_OBJECT_SENSOR = CN()
_C.TASK.GRIPPED_OBJECT_SENSOR.TYPE = "GrippedObjectSensor"
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT POSITIONS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_POSITION = CN()
_C.TASK.OBJECT_POSITION.TYPE = "ObjectPosition"
_C.TASK.OBJECT_POSITION.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_POSITION.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # REARRANGEMENT TASK ALL OBJECT GOALS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_GOAL = CN()
_C.TASK.OBJECT_GOAL.TYPE = "ObjectGoal"
_C.TASK.OBJECT_GOAL.GOAL_FORMAT = "POLAR"
_C.TASK.OBJECT_GOAL.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_TO_GOAL_DISTANCE = CN()
_C.TASK.OBJECT_TO_GOAL_DISTANCE.TYPE = "ObjectToGoalDistance"
# -----------------------------------------------------------------------------
# # OBJECT_DISTANCE_FROM_AGENT MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.AGENT_TO_OBJECT_DISTANCE = CN()
_C.TASK.AGENT_TO_OBJECT_DISTANCE.TYPE = "AgentToObjectDistance"




def merge_sim_episode_with_object_config(
    sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config = merge_sim_episode_config(sim_config, episode)
    sim_config.defrost()
    sim_config.objects = [episode.objects]
    sim_config.freeze()

    return sim_config


@registry.register_task(name="RearrangementTask-v0")
class RearrangementTask(NavigationTask):
    r"""Embodied Rearrangement Task
    Goal: An agent must place objects at their corresponding goal position.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def overwrite_sim_config(self, sim_config, episode):
        return merge_sim_episode_with_object_config(sim_config, episode)
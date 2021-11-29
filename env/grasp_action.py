import habitat_sim
import attr
from habitat.core.registry import registry
from typing import List, Any
import magnum as mn
# @title Define a Grab/Release action and create a new action space.
# @markdown Each new action is defined by a `ActionSpec` and an `ActuationSpec`. `ActionSpec` is mapping between the action name and its corresponding `ActuationSpec`. `ActuationSpec` contains all the necessary specifications required to define the action.

from habitat.config.default import _C, CN
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat_sim.agent.controls.controls import ActuationSpec
from habitat_sim.physics import MotionType
# @title RayCast utility to implement Grab/Release Under Cross-Hair Action
# @markdown Cast a ray in the direction of crosshair from the camera and check if it collides with another object within a certain distance threshold


def raycast(sim, sensor_name, crosshair_pos=(128, 128), max_distance=2.0):
    r"""Cast a ray in the direction of crosshair and check if it collides
    with another object within a certain distance threshold
    :param sim: Simulator object
    :param sensor_name: name of the visual sensor to be used for raycasting
    :param crosshair_pos: 2D coordiante in the viewport towards which the
        ray will be cast
    :param max_distance: distance threshold beyond which objects won't
        be considered
    """
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera
    center_ray = render_camera.unproject(mn.Vector2i(crosshair_pos))

    raycast_results = sim.cast_ray(center_ray, max_distance=max_distance)

    closest_object = -1
    closest_dist = 1000.0
    if raycast_results.has_hits():
        for hit in raycast_results.hits:
            if hit.ray_distance < closest_dist:
                closest_dist = hit.ray_distance
                closest_object = hit.object_id

    return closest_object


#from habitat_sim.physics import MotionType


# @markdown For instance, `GrabReleaseActuationSpec` contains the following:
# @markdown - `visual_sensor_name` defines which viewport (rgb, depth, etc) to to use to cast the ray.
# @markdown - `crosshair_pos` stores the position in the viewport through which the ray passes. Any object which intersects with this ray can be grabbed by the agent.
# @markdown - `amount` defines a distance threshold. Objects which are farther than the treshold cannot be picked up by the agent.
@attr.s(auto_attribs=True, slots=True)
class GrabReleaseActuationSpec(ActuationSpec):
    visual_sensor_name: str = "rgb"
    crosshair_pos: List[int] = [128, 128]
    amount: float = 2.0


# @markdown Then, we extend the `HabitatSimV1ActionSpaceConfiguration` to add the above action into the agent's action space. `ActionSpaceConfiguration` is a mapping between action name and the corresponding `ActionSpec`
@registry.register_action_space_configuration(name="RearrangementActions-v0")
class RearrangementSimV0ActionSpaceConfiguration(
    HabitatSimV1ActionSpaceConfiguration
):
    def __init__(self, config):
        super().__init__(config)
        if not HabitatSimActions.has_action("GRAB_RELEASE"):
            HabitatSimActions.extend_action_space("GRAB_RELEASE")

    def get(self):
        config = super().get()
        new_config = {
            HabitatSimActions.GRAB_RELEASE: habitat_sim.ActionSpec(
                "grab_or_release_object_under_crosshair",
                GrabReleaseActuationSpec(
                    visual_sensor_name=self.config.VISUAL_SENSOR,
                    crosshair_pos=self.config.CROSSHAIR_POS,
                    amount=self.config.GRAB_DISTANCE,
                ),
            )
        }

        config.update(new_config)

        return config


# @markdown Finally, we extend `SimualtorTaskAction` which tells the simulator which action to call when a named action ('GRAB_RELEASE' in this case) is predicte by the agent's policy.
@registry.register_task_action
class GrabOrReleaseAction(SimulatorTaskAction):
    def step(self, *args: Any, **kwargs: Any):
        r"""This method is called from ``Env`` on each ``step``."""
        return self._sim.step(HabitatSimActions.GRAB_RELEASE)


_C.TASK.ACTIONS.GRAB_RELEASE = CN()
_C.TASK.ACTIONS.GRAB_RELEASE.TYPE = "GrabOrReleaseAction"
_C.SIMULATOR.CROSSHAIR_POS = [128, 160]
_C.SIMULATOR.GRAB_DISTANCE = 2.0
_C.SIMULATOR.VISUAL_SENSOR = "rgb"
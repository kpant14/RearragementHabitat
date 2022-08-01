from habitat.core.utils import DatasetFloatJSONEncoder, not_none_validator
from typing import Optional, Dict, List
import json
from habitat.config.default import CN, Config
import os
import attr
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.tasks.nav.nav import NavigationEpisode

@attr.s(auto_attribs=True, kw_only=True)
class RearrangementSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """
    position: List[float] = attr.ib(default=None, validator=not_none_validator)
    rotation: List[float] = attr.ib(default=None, validator=not_none_validator)
    info: Optional[Dict[str, str]] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementObjectSpec(RearrangementSpec):
    r"""Object specifications that capture position of each object in the scene,
    the associated object template.
    """
    object_handle: str = attr.ib(default=None, validator=not_none_validator)
    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_template: Optional[str] = attr.ib(
        default=None
    )

class PickupSpec:
    r"""Specifications that capture a particular position of final position
    or initial position of the object.
    """
    pickup_order: List[int] = attr.ib(default=None, validator=not_none_validator)
    pickup_order_tdmap: List[int] = attr.ib(default=None, validator=not_none_validator)
    pickup_order_l2dist: List[int] = attr.ib(default=None, validator=not_none_validator)
    


@attr.s(auto_attribs=True, kw_only=True)
class RearrangementEpisode(NavigationEpisode):
    r"""Specification of episode that includes initial position and rotation
    of agent, all goal specifications, all object specifications

    Args:
        episode_id: id of episode in the dataset
        scene_id: id of scene inside the simulator.
        start_position: numpy ndarray containing 3 entries for (x, y, z).
        start_rotation: numpy ndarray with 4 entries for (x, y, z, w)
            elements of unit quaternion (versor) representing agent 3D
            orientation.
        goal: object's goal position and rotation
        object: object's start specification defined with object type,
            position, and rotation.
    """
    objects: RearrangementObjectSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    goals: RearrangementSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    pickup_order: PickupSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    pickup_order_tdmap: PickupSpec = attr.ib(
        default=None, validator=not_none_validator
    )
    pickup_order_l2dist: PickupSpec = attr.ib(
        default=None, validator=not_none_validator
    )

@registry.register_dataset(name="RearrangementDataset-v0")
class RearrangementDatasetV0(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads Rearrangement dataset."""
    episodes: List[RearrangementEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def to_json(self) -> str:
        result = DatasetFloatJSONEncoder().encode(self)
        return result

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        for i, episode in enumerate(deserialized["episodes"]):
            rearrangement_episode = RearrangementEpisode(**episode)
            rearrangement_episode.episode_id = str(i)

            if scenes_dir is not None:
                if rearrangement_episode.scene_id.startswith(
                    DEFAULT_SCENE_PATH_PREFIX
                ):
                    rearrangement_episode.scene_id = (
                        rearrangement_episode.scene_id[
                            len(DEFAULT_SCENE_PATH_PREFIX) :
                        ]
                    )

                rearrangement_episode.scene_id = os.path.join(
                    scenes_dir, rearrangement_episode.scene_id
                )
            for i, obj in enumerate(rearrangement_episode.objects):
                idx = obj["object_handle"]
                if type(idx) is not str:
                    template = rearrangement_episode.object_templates[idx]
                    obj["object_handle"] = template["object_handle"]
                rearrangement_episode.objects[i] = RearrangementObjectSpec(**obj)

            for i, goal in enumerate(rearrangement_episode.goals):
                rearrangement_episode.goals[i] = RearrangementSpec(**goal)

            self.episodes.append(rearrangement_episode)           
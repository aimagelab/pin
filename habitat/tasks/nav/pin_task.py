import glob
import json
from typing import Any, List, Optional
import attr
import os
import imageio
import numpy as np

from omegaconf import DictConfig
from gym import Space, spaces
from habitat.core.registry import registry
from habitat.core.dataset import SceneState
from habitat.core.simulator import RGBSensor, VisualObservation
from habitat.tasks.nav.nav import NavigationEpisode, NavigationTask
from habitat.tasks.nav.object_nav_task import ObjectGoalNavEpisode, ReplayActionSpec
from habitat.core.utils import not_none_validator

from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import (
    RGBSensor,
    VisualObservation,
)
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import ObjectGoal

try:
    from habitat.datasets.pin.pin import (
        PINDatasetV1,
    )
except ImportError:
    pass

from typing import Any

from habitat.core.registry import registry
from habitat.tasks.nav.nav import NavigationTask

@attr.s(auto_attribs=True, kw_only=True)
class PINEpisodeV1(NavigationEpisode):
    r"""Personalized Navigation Episode"""
    # object_index: Optional[int]
    object_category: Optional[List[str]] = None
    object_id: Optional[str] = None
    distractors: List[Any] = []

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals"""
        return (
            f"{os.path.basename(self.scene_id)}_{self.object_category}_{self.object_id}"
        )
        

@registry.register_task(name="PIN-v1")
class PINTaskV1(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
    Used to explicitly state a type of the task in config.
    """
    _is_episode_active: bool
    _prev_action: int

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._is_episode_active = False

    def overwrite_sim_config(self, sim_config, episode):
        super().overwrite_sim_config(sim_config, episode)
        return sim_config

    def _check_episode_is_active(self, action, *args: Any, **kwargs: Any) -> bool:
        return not getattr(self, "is_stop_called", False)


@registry.register_sensor
class PINGoalSensor(RGBSensor):
    """A sensor for instance-based image goal specification used by the
    InstanceImageGoal Navigation task. Image goals are rendered according to
    camera parameters (resolution, HFOV, extrinsics) specified by the dataset.

    Args:
        sim: a reference to the simulator for rendering instance image goals.
        config: a config for the InstanceImageGoalSensor sensor.
        dataset: a Instance Image Goal navigation dataset that contains a
        dictionary mapping goal IDs to instance image goals.
    """

    cls_uuid: str = "pin_goal"
    _current_reference_images: Optional[VisualObservation]
    _current_reference_texts: Optional[List[str]]
    _current_episode_id: Optional[str]

    def __init__(
        self,
        sim,
        config: "DictConfig",
        dataset: "PINDatasetV1",
        *args: Any,
        **kwargs: Any,
    ):
        from habitat.datasets.pin.pin import (
            PINDatasetV1,
        )

        assert isinstance(
            dataset, PINDatasetV1
        ), "Provided dataset needs to be PINDatasetV1"

        self._dataset = dataset
        self._sim = sim
        super().__init__(config=config)
        self._current_episode_id = None
        self._current_reference_images = None
        self._current_reference_texts = None
        self._reference_texts = json.load(
            open(
                os.path.join(
                    "/".join(self.config.object_images_path.split("/")[:-1]),
                    "object_descriptions.json",
                )
            )
        )

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> Space:
        H, W = self.config.H, self.config.W
        return spaces.Box(low=0, high=255, shape=(H, W, 4), dtype=np.uint8)

    def _get_reference_images(self, obj_id, category) -> VisualObservation:
        full_path = f"{self.config.object_images_path}/{category}/{obj_id}_*.png"
        imgs = []
        for img_path in glob.glob(full_path):
            img = imageio.imread(img_path)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        return imgs

    def _get_reference_texts(self, obj_id) -> VisualObservation:
        return self._reference_texts[obj_id]

    def get_observation(
        self,
        *args: Any,
        episode: PINEpisodeV1,
        **kwargs: Any,
    ) -> Optional[VisualObservation]:
        if len(episode.goals) == 0:
            logger.error(f"No goal specified for episode {episode.episode_id}.")
            return None
        if not isinstance(episode.goals[0], ObjectGoal):
            logger.error(f"Goal should be ObjectGoal, episode {episode.episode_id}.")
            return None

        episode_uniq_id = f"{episode.scene_id} {episode.episode_id}"
        if episode_uniq_id == self._current_episode_id:
            return self._current_reference_images, self._current_reference_texts

        obj_id = episode.goals[0].object_id
        category = episode.goals[0].object_category
        self._current_reference_images = self._get_reference_images(obj_id, category)
        self._current_reference_texts = self._get_reference_texts(obj_id)
        self._current_episode_id = episode_uniq_id

        return self._current_reference_images, self._current_reference_texts

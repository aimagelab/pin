#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import gym
import magnum
import numba
import numpy as np
from gym import spaces

from habitat.config import read_write
from habitat.core.dataset import BaseEpisode, Dataset, Episode, EpisodeIterator
from habitat.core.embodied_task import EmbodiedTask, Metrics
from habitat.core.simulator import Observations, Simulator
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.nav.multi_object_nav_task import MultiObjectGoalNavEpisode
from habitat.tasks.nav.pin_task import PINEpisodeV1
from habitat.tasks.registration import make_task
from habitat.utils import profiling_wrapper
from habitat_sim.gfx import NO_LIGHT_KEY

if TYPE_CHECKING:
    from omegaconf import DictConfig


class Env:
    r"""Fundamental environment class for :ref:`habitat`.

    :data observation_space: ``SpaceDict`` object corresponding to sensor in
        sim and task.
    :data action_space: ``gym.space`` object corresponding to valid actions.

    All the information  needed for working on embodied task with simulator
    is abstracted inside :ref:`Env`. Acts as a base for other derived
    environment classes. :ref:`Env` consists of three major components:
    ``dataset`` (`episodes`), ``simulator`` (:ref:`sim`) and :ref:`task` and
    connects all the three components together.
    """

    observation_space: spaces.Dict
    action_space: spaces.Dict
    _config: "DictConfig"
    _dataset: Optional[Dataset[Episode]]
    number_of_episodes: Optional[int]
    _current_episode: Optional[Episode]
    _episode_iterator: Optional[Iterator[Episode]]
    _sim: Simulator
    _task: EmbodiedTask
    _max_episode_seconds: int
    _max_episode_steps: int
    _elapsed_steps: int
    _episode_start_time: Optional[float]
    _episode_over: bool
    _episode_from_iter_on_reset: bool
    _episode_force_changed: bool

    def __init__(
        self, config: "DictConfig", dataset: Optional[Dataset[Episode]] = None
    ) -> None:
        """Constructor

        :param config: config for the environment. Should contain id for
            simulator and ``task_name`` which are passed into ``make_sim`` and
            ``make_task``.
        :param dataset: reference to dataset for task instance level
            information. Can be defined as :py:`None` in which case
            ``_episodes`` should be populated from outside.
        """

        if "habitat" in config:
            config = config.habitat
        self._config = config
        self._dataset = dataset
        if self._dataset is None and config.dataset.type:
            self._dataset = make_dataset(
                id_dataset=config.dataset.type, config=config.dataset
            )

        self._current_episode = None
        self._episode_iterator = None
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        # load the first scene if dataset is present
        if self._dataset:
            assert (
                len(self._dataset.episodes) > 0
            ), "dataset should have non-empty episodes list"
            self._setup_episode_iterator()
            self.current_episode = next(self.episode_iterator)
            with read_write(self._config):
                self._config.simulator.scene_dataset = (
                    self.current_episode.scene_dataset_config
                )
                self._config.simulator.scene = self.current_episode.scene_id

            self.number_of_episodes = len(self.episodes)
        else:
            self.number_of_episodes = None

        self._sim = make_sim(
            id_sim=self._config.simulator.type, config=self._config.simulator
        )

        self._task = make_task(
            self._config.task.type,
            config=self._config.task,
            sim=self._sim,
            dataset=self._dataset,
        )
        self.observation_space = spaces.Dict(
            {
                **self._sim.sensor_suite.observation_spaces.spaces,
                **self._task.sensor_suite.observation_spaces.spaces,
            }
        )
        self.action_space = self._task.action_space
        self._max_episode_seconds = self._config.environment.max_episode_seconds
        self._max_episode_steps = self._config.environment.max_episode_steps
        self._elapsed_steps = 0
        self._episode_start_time: Optional[float] = None
        self._episode_over = False

    def _setup_episode_iterator(self):
        assert self._dataset is not None
        iter_option_dict = {
            k.lower(): v for k, v in self._config.environment.iterator_options.items()
        }
        iter_option_dict["seed"] = self._config.seed
        self._episode_iterator = self._dataset.get_episode_iterator(**iter_option_dict)

    @property
    def current_episode(self) -> Episode:
        assert self._current_episode is not None
        return self._current_episode

    @current_episode.setter
    def current_episode(self, episode: Episode) -> None:
        self._current_episode = episode
        # This allows the current episode to be set here
        # and then reset be called without the episode changing
        self._episode_from_iter_on_reset = False
        self._episode_force_changed = True

    @property
    def episode_iterator(self) -> Iterator[Episode]:
        return self._episode_iterator

    @episode_iterator.setter
    def episode_iterator(self, new_iter: Iterator[Episode]) -> None:
        self._episode_iterator = new_iter
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def episodes(self) -> List[Episode]:
        return self._dataset.episodes if self._dataset else cast(List[Episode], [])

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        assert len(episodes) > 0, "Environment doesn't accept empty episodes list."
        assert (
            self._dataset is not None
        ), "Environment must have a dataset to set episodes"
        self._dataset.episodes = episodes
        self._setup_episode_iterator()
        self._current_episode = None
        self._episode_force_changed = True
        self._episode_from_iter_on_reset = True

    @property
    def sim(self) -> Simulator:
        return self._sim

    @property
    def episode_start_time(self) -> Optional[float]:
        return self._episode_start_time

    @property
    def episode_over(self) -> bool:
        return self._episode_over

    @property
    def task(self) -> EmbodiedTask:
        return self._task

    @property
    def _elapsed_seconds(self) -> float:
        assert (
            self._episode_start_time
        ), "Elapsed seconds requested before episode was started."
        return time.time() - self._episode_start_time

    def get_metrics(self) -> Metrics:
        return self._task.measurements.get_metrics()

    def _past_limit(self) -> bool:
        return (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ) or (
            self._max_episode_seconds != 0
            and self._max_episode_seconds <= self._elapsed_seconds
        )

    def _reset_stats(self) -> None:
        self._episode_start_time = time.time()
        self._elapsed_steps = 0
        self._episode_over = False

    def reset(self) -> Observations:
        r"""Resets the environments and returns the initial observations.

        :return: initial observations from the environment.
        """
        self._reset_stats()

        # Delete the shortest path cache of the current episode
        # Caching it for the next time we see this episode isn't really worth
        # it
        if self._current_episode is not None:
            self._current_episode._shortest_path_cache = None

        if self._episode_iterator is not None and self._episode_from_iter_on_reset:
            self._current_episode = next(self._episode_iterator)

        # This is always set to true after a reset that way
        # on the next reset an new episode is taken (if possible)
        self._episode_from_iter_on_reset = True
        self._episode_force_changed = False

        assert self._current_episode is not None, "Reset requires an episode"
        self.reconfigure(self._config)

        if hasattr(self._dataset, "category_to_task_category_id"):
            rigid_obj_mgr = self._sim.get_rigid_object_manager()
            self.object_to_dataset_mapping = self._dataset.category_to_task_category_id
            # Remove existing objects from last episode
            rigid_obj_mgr.remove_all_objects()

            # Insert current episode objects
            obj_path = self._config.task.objects_path
            obj_path = self._config.task.objects_path.replace(
                "{split}", self._config.dataset.split
            )

            obj_templates_mgr = self._sim.get_object_template_manager()
            if os.path.isdir(obj_path):
                obj_templates_mgr.load_configs(obj_path, True)
            # print(f"Number of loaded objects: {obj_templates_mgr.get_num_templates()}")

            if type(self.current_episode) == PINEpisodeV1:
                object_id = str(self.current_episode.goals[0].object_id)
                current_goal = self.current_episode.goals[0].object_category
                dataset_index = self.object_to_dataset_mapping[current_goal]
                obj_handle_list = obj_templates_mgr.get_template_handles(object_id)[0]
                object_box = rigid_obj_mgr.add_object_by_template_handle(
                    obj_handle_list, light_setup_key=NO_LIGHT_KEY, 
                )
        
                pos = np.array(self.current_episode.goals[0].position)
                object_box.semantic_id = 100000
                object_box.translation = np.array(pos)
                
                if hasattr(self._config.task, "include_distractors") and self._config.task.include_distractors:
                    for i in range(len(self.current_episode.distractors)):
                        current_distractor = self.current_episode.distractors[i].object_category
                        if current_distractor == current_goal and hasattr(self._config.task, "exclude_cat_distractors") and self._config.task.exclude_cat_distractors:
                            continue
                        object_id = self.current_episode.distractors[i].object_id

                        dataset_index = self.object_to_dataset_mapping[current_distractor]

                        try:
                            obj_handle_list = obj_templates_mgr.get_template_handles(
                                object_id
                            )[0]
                            object_box = rigid_obj_mgr.add_object_by_template_handle(
                                obj_handle_list, light_setup_key=NO_LIGHT_KEY
                            )
                            pos = np.array(
                                self.current_episode.distractors[i].position
                            )
                            if current_distractor == current_goal:
                                object_box.semantic_id = 100001
                            else:
                                object_box.semantic_id = 100002
                            object_box.translation = np.array(pos)
                        except:
                            with open("logs/pin_missing_objects.txt", "a") as f:
                                f.write(f"{object_id}\n")

            elif type(self.current_episode) == MultiObjectGoalNavEpisode:  # multion
                for i in range(len(self.current_episode.goals)):
                    current_goal = self.current_episode.goals[i].object_category
                    object_id = str(self.current_episode.goals[i].object_id)
                    current_goal = self.current_episode.goals[i].object_category
                    dataset_index = self.object_to_dataset_mapping[current_goal]
                    obj_handle_list = obj_templates_mgr.get_template_handles(object_id)[
                        0
                    ]
                    object_box = rigid_obj_mgr.add_object_by_template_handle(
                        obj_handle_list
                    )
                    obj_node = object_box.root_scene_node
                    obj_bb = obj_node.cumulative_bb
                    jj = obj_bb.back_bottom_left
                    jj = [jj[0], jj[2], jj[1]]
                    diff = np.array(self.current_episode.goals[i].position)
                    diff2 = diff - jj
                    diff2[2] += jj[2] * 2
                    diff2[1] += 0.05
                    object_box.semantic_id = dataset_index
                    object_box.translation = np.array(diff2)
                    object_box.rotate_x(magnum.Rad(-1.5708))

                if hasattr(self._config.task, "include_distractors") and self._config.task.include_distractors:
                    for i in range(len(self.current_episode.distractors)):
                        current_distractor = self.current_episode.distractors[
                            i
                        ].object_category
                        object_id = self.current_episode.distractors[i].object_id

                        dataset_index = self.object_to_dataset_mapping[
                            current_distractor
                        ]

                        try:
                            obj_handle_list = obj_templates_mgr.get_template_handles(
                                object_id
                            )[0]
                            object_box = rigid_obj_mgr.add_object_by_template_handle(
                                obj_handle_list
                            )
                            obj_node = object_box.root_scene_node
                            obj_bb = obj_node.cumulative_bb
                            jj = obj_bb.back_bottom_left
                            jj = [jj[0], jj[2], jj[1]]
                            diff = np.array(
                                self.current_episode.distractors[i].position
                            )
                            diff2 = diff - jj
                            diff2[2] += jj[2] * 2
                            diff2[1] += 0.05
                            object_box.semantic_id = dataset_index
                            object_box.translation = np.array(diff2)
                            object_box.rotate_x(magnum.Rad(-1.5708))
                        except:
                            with open("logs/multion_missing_objects.txt", "a") as f:
                                f.write(f"{object_id}\n")

            else:
                print("No additional objects added...")
                pass

        observations = self.task.reset(episode=self.current_episode)
        self._task.measurements.reset_measures(
            episode=self.current_episode,
            task=self.task,
            observations=observations,
        )

        return observations

    def _update_step_stats(self) -> None:
        self._elapsed_steps += 1
        self._episode_over = not self._task.is_episode_active

        if self._past_limit():
            self._episode_over = True

        if self.episode_iterator is not None and isinstance(
            self.episode_iterator, EpisodeIterator
        ):
            self.episode_iterator.step_taken()

    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        r"""Perform an action in the environment and return observations.

        :param action: action (belonging to :ref:`action_space`) to be
            performed inside the environment. Action is a name or index of
            allowed task's action and action arguments (belonging to action's
            :ref:`action_space`) to support parametrized and continuous
            actions.
        :return: observations after taking action in environment.
        """

        assert (
            self._episode_start_time is not None
        ), "Cannot call step before calling reset"
        assert (
            self._episode_over is False
        ), "Episode over, call reset before calling step"
        assert (
            not self._episode_force_changed
        ), "Episode was changed either by setting current_episode or changing the episodes list. Call reset before stepping the environment again."

        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            action = {"action": action}

        observations = self.task.step(action=action, episode=self.current_episode)

        self._task.measurements.update_measures(
            episode=self.current_episode,
            action=action,
            task=self.task,
            observations=observations,
        )

        ##Terminates episode if wrong found is called
        if "sub_success" in self.task.measurements.measures:
            if (
                self.task.is_found_called == True
                and self.task.measurements.measures["sub_success"].get_metric() == 0
            ):
                self.task._is_episode_active = False

        ##Terminates episode if all goals are found
            if self.task.is_found_called == True and self.task.current_goal_index == len(
                self.current_episode.goals
            ):
                self.task._is_episode_active = False

        self._update_step_stats()

        return observations

    @staticmethod
    @numba.njit
    def _seed_numba(seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        self._seed_numba(seed)
        self._sim.seed(seed)
        self._task.seed(seed)

    def reconfigure(self, config: "DictConfig") -> None:
        self._config = config

        with read_write(self._config):
            self._config.simulator = self._task.overwrite_sim_config(
                self._config.simulator, self.current_episode
            )

        self._sim.reconfigure(self._config.simulator)

    def render(self, mode="rgb") -> np.ndarray:
        return self._sim.render(mode)

    def close(self) -> None:
        self._sim.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RLEnv(gym.Env):
    r"""Reinforcement Learning (RL) environment class which subclasses ``gym.Env``.

    This is a wrapper over :ref:`Env` for RL users. To create custom RL
    environments users should subclass `RLEnv` and define the following
    methods: :ref:`get_reward_range()`, :ref:`get_reward()`,
    :ref:`get_done()`, :ref:`get_info()`.

    As this is a subclass of ``gym.Env``, it implements `reset()` and
    `step()`.
    """

    _env: Env

    def __init__(self, config: "DictConfig", dataset: Optional[Dataset] = None) -> None:
        """Constructor

        :param config: config to construct :ref:`Env`
        :param dataset: dataset to construct :ref:`Env`.
        """
        if "habitat" in config:
            config = config.habitat
        self._core_env_config = config
        self._env = Env(config, dataset)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.number_of_episodes = self._env.number_of_episodes
        self.reward_range = self.get_reward_range()

    @property
    def config(self) -> "DictConfig":
        return self._core_env_config

    @property
    def habitat_env(self) -> Env:
        return self._env

    @property
    def episodes(self) -> List[Episode]:
        return self._env.episodes

    @episodes.setter
    def episodes(self, episodes: List[Episode]) -> None:
        self._env.episodes = episodes

    def current_episode(self, all_info: bool = False) -> BaseEpisode:
        """
        Returns the current episode of the environment.
        :param all_info: If true, all of the information in the episode
        will be provided. Otherwise, only episode_id and scene_id will
        be included
        :return: The BaseEpisode object for the current episode
        """
        if all_info:
            return self._env.current_episode
        else:
            return BaseEpisode(
                episode_id=self._env.current_episode.episode_id,
                scene_id=self._env.current_episode.scene_id,
            )

    @profiling_wrapper.RangeContext("RLEnv.reset")
    def reset(self) -> Observations:
        return self._env.reset()

    def get_reward_range(self):
        r"""Get min, max range of reward.

        :return: :py:`[min, max]` range of reward.
        """
        raise NotImplementedError

    def get_reward(self, observations: Observations) -> Any:
        r"""Returns reward after action has been performed.

        :param observations: observations from simulator and task.
        :return: reward after performing the last action.

        This method is called inside the :ref:`step()` method.
        """
        raise NotImplementedError

    def get_done(self, observations: Observations) -> bool:
        r"""Returns boolean indicating whether episode is done after performing
        the last action.

        :param observations: observations from simulator and task.
        :return: done boolean after performing the last action.

        This method is called inside the step method.
        """
        raise NotImplementedError

    def get_info(self, observations) -> Dict[Any, Any]:
        r"""..

        :param observations: observations from simulator and task.
        :return: info after performing the last action.
        """
        raise NotImplementedError

    @profiling_wrapper.RangeContext("RLEnv.step")
    def step(self, *args, **kwargs) -> Tuple[Observations, Any, bool, dict]:
        r"""Perform an action in the environment.

        :return: :py:`(observations, reward, done, info)`
        """

        observations = self._env.step(*args, **kwargs)
        reward = self.get_reward(observations)
        done = self.get_done(observations)
        info = self.get_info(observations)

        return observations, reward, done, info

    def seed(self, seed: Optional[int] = None) -> None:
        self._env.seed(seed)

    def render(self, mode: str = "rgb") -> np.ndarray:
        return self._env.render(mode)

    def close(self) -> None:
        self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

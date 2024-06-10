# modified from: https://github.com/allenai/allenact/blob/main/projects/objectnav_baselines/experiments/objectnav_thor_base.py

import glob
import os
import platform
from abc import ABC
from math import ceil
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import os
from typing import List

import habitat
from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_BASE,
    HABITAT_CONFIGS_DIR,
)
from omegaconf import DictConfig as Config
import gym
import numpy as np
from omegaconf import read_write
import torch
from allenact.base_abstractions.experiment_config import MachineParams
from allenact.base_abstractions.preprocessor import SensorPreprocessorGraph
from allenact.base_abstractions.sensor import ExpertActionSensor, SensorSuite
from allenact.base_abstractions.task import TaskSampler
from allenact.utils.experiment_utils import evenly_distribute_count_into_bins
from allenact.utils.system import get_logger
from allenact_plugins.habitat_plugin.habitat_constants import (
    HABITAT_CONFIGS_DIR,
    HABITAT_DATASETS_DIR,
    HABITAT_SCENE_DATASETS_DIR,
)
from allenact_plugins.habitat_plugin.habitat_task_samplers import ObjectNavTaskSampler
from allenact_plugins.habitat_plugin.habitat_utils import (
    get_habitat_config,
)
from allenact_plugins.robothor_plugin.robothor_tasks import ObjectNavTask
from abc import ABC
from typing import Optional, Sequence, Union

from allenact.base_abstractions.experiment_config import ExperimentConfig
from allenact.base_abstractions.preprocessor import Preprocessor
from allenact.base_abstractions.sensor import Sensor
from allenact.utils.experiment_utils import Builder
import numpy as np

from utils.cow_utils.vision_sensor import VISION_SENSOR_TO_HABITAT_LABEL


def construct_env_configs(
    config: Config,
    allow_scene_repeat: bool = False,
) -> List[Config]:
    """Create list of Habitat Configs for training on multiple processes To
    allow better performance, dataset are split into small ones for each
    individual env, grouped by scenes.

    # Parameters

    config : configs that contain num_processes as well as information
             necessary to create individual environments.
    allow_scene_repeat: if `True` and the number of distinct scenes
        in the dataset is less than the total number of processes this will
        result in scenes being repeated across processes. If `False`, then
        if the total number of processes is greater than the number of scenes,
        this will result in a RuntimeError exception being raised.

    # Returns

    List of Configs, one for each process.
    """

    num_processes = config.num_environments
    configs = []
    dataset = habitat.make_dataset(config.habitat.dataset.type)
    scenes = dataset.get_scenes_to_load(config.habitat.dataset)

    if len(scenes) > 0:
        if len(scenes) < num_processes:
            if not allow_scene_repeat:
                raise RuntimeError(
                    "reduce the number of processes as there aren't enough number of scenes."
                )
            else:
                scenes = (scenes * (1 + (num_processes // len(scenes))))[:num_processes]

    scene_splits: List[List] = [[] for _ in range(num_processes)]
    for idx, scene in enumerate(scenes):
        scene_splits[idx % len(scene_splits)].append(scene)

    assert sum(map(len, scene_splits)) == len(scenes)

    for i in range(num_processes):
        task_config = config.copy()
        with read_write(task_config):
            if len(scenes) > 0:
                task_config.habitat.dataset.content_scenes = scene_splits[i]

            if (
                type(config.simulator_gpu_id) == int
                or len(config.simulator_gpu_id) == 0
            ):
                task_config.habitat.simulator.habitat_sim_v0.gpu_device_id = -1
            else:
                task_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
                    config.simulator_gpu_id[i % len(config.simulator_gpu_id)]
                )
        configs.append(task_config.copy())

    return configs


def construct_env_configs_mp3d(config: Config) -> List[Config]:
    r"""Create list of Habitat Configs for training on multiple processes
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    Args:
        config: configs that contain num_processes as well as information
        necessary to create individual environments.
    Returns:
        List of Configs, one for each process
    """

    config.freeze()
    num_processes = config.NUM_PROCESSES
    configs = []
    # dataset = habitat.make_dataset(config.DATASET.TYPE)
    # scenes = dataset.get_scenes_to_load(config.DATASET)

    if num_processes == 1:
        scene_splits = [["pRbA3pwrgk9"]]
    else:
        small = [
            "rPc6DW4iMge",
            "e9zR4mvMWw7",
            "uNb9QFRL6hY",
            "qoiz87JEwZ2",
            "sKLMLpTHeUy",
            "s8pcmisQ38h",
            "759xd9YjKW5",
            "XcA2TqTSSAj",
            "SN83YJsR3w2",
            "8WUmhLawc2A",
            "JeFG25nYj2p",
            "17DRP5sb8fy",
            "Uxmj2M2itWa",
            "XcA2TqTSSAj",
            "SN83YJsR3w2",
            "8WUmhLawc2A",
            "JeFG25nYj2p",
            "17DRP5sb8fy",
            "Uxmj2M2itWa",
            "D7N2EKCX4Sj",
            "b8cTxDM8gDG",
            "sT4fr6TAbpF",
            "S9hNv5qa7GM",
            "82sE5b5pLXE",
            "pRbA3pwrgk9",
            "aayBHfsNo7d",
            "cV4RVeZvu5T",
            "i5noydFURQK",
            "YmJkqBEsHnH",
            "jh4fc5c5qoQ",
            "VVfe2KiqLaN",
            "29hnd4uzFmX",
            "Pm6F8kyY3z2",
            "JF19kD82Mey",
            "GdvgFV5R1Z5",
            "HxpKQynjfin",
            "vyrNrziPKCB",
        ]
        med = [
            "V2XKFyX4ASd",
            "VFuaQ6m2Qom",
            "ZMojNkEp431",
            "5LpN3gDmAk7",
            "r47D5H71a5s",
            "ULsKaCPVFJR",
            "E9uDoFAP3SH",
            "kEZ7cmS4wCh",
            "ac26ZMwG7aT",
            "dhjEzFoUFzH",
            "mJXqzFtmKg4",
            "p5wJjkQkbXX",
            "Vvot9Ly1tCj",
            "EDJbREhghzL",
            "VzqfbhrpDEA",
            "7y3sRwLe3Va",
        ]

        scene_splits = [[] for _ in range(config.NUM_PROCESSES)]
        distribute(
            small,
            scene_splits,
            num_gpus=8,
            procs_per_gpu=3,
            proc_offset=1,
            scenes_per_process=2,
        )
        distribute(
            med,
            scene_splits,
            num_gpus=8,
            procs_per_gpu=3,
            proc_offset=0,
            scenes_per_process=1,
        )

        # gpu0 = [['pRbA3pwrgk9', '82sE5b5pLXE', 'S9hNv5qa7GM'],
        #         ['Uxmj2M2itWa', '17DRP5sb8fy', 'JeFG25nYj2p'],
        #         ['5q7pvUzZiYa', '759xd9YjKW5', 's8pcmisQ38h'],
        #         ['e9zR4mvMWw7', 'rPc6DW4iMge', 'vyrNrziPKCB']]
        # gpu1 = [['sT4fr6TAbpF', 'b8cTxDM8gDG', 'D7N2EKCX4Sj'],
        #         ['8WUmhLawc2A', 'SN83YJsR3w2', 'XcA2TqTSSAj'],
        #         ['sKLMLpTHeUy', 'qoiz87JEwZ2', 'uNb9QFRL6hY'],
        #         ['V2XKFyX4ASd', 'VFuaQ6m2Qom', 'ZMojNkEp431']]
        # gpu2 = [['5LpN3gDmAk7', 'r47D5H71a5s', 'ULsKaCPVFJR', 'E9uDoFAP3SH'],
        #         ['VVfe2KiqLaN', 'jh4fc5c5qoQ', 'YmJkqBEsHnH'],  # small
        #         ['i5noydFURQK', 'cV4RVeZvu5T', 'aayBHfsNo7d']]  # small
        # gpu3 = [['kEZ7cmS4wCh', 'ac26ZMwG7aT', 'dhjEzFoUFzH'],
        #         ['mJXqzFtmKg4', 'p5wJjkQkbXX', 'Vvot9Ly1tCj']]
        # gpu4 = [['EDJbREhghzL', 'VzqfbhrpDEA', '7y3sRwLe3Va'],
        #         ['ur6pFq6Qu1A', 'PX4nDJXEHrG', 'PuKPg4mmafe']]
        # gpu5 = [['r1Q1Z4BcV1o', 'gTV8FGcVJC9', '1pXnuDYAj8r'],
        #         ['JF19kD82Mey', 'Pm6F8kyY3z2', '29hnd4uzFmX']]  # small
        # gpu6 = [['VLzqgDo317F', '1LXtFkjw3qL'],
        #         ['HxpKQynjfin', 'gZ6f7yhEvPG', 'GdvgFV5R1Z5']]  # small
        # gpu7 = [['D7G3Y4RVNrH', 'B6ByNegPMKs']]
        #
        # scene_splits = gpu0 + gpu1 + gpu2 + gpu3 + gpu4 + gpu5 + gpu6 + gpu7

    for i in range(num_processes):
        task_config = config.clone()
        task_config.defrost()
        task_config.DATASET.CONTENT_SCENES = scene_splits[i]

        task_config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = config.SIMULATOR_GPU_IDS[
            i % len(config.SIMULATOR_GPU_IDS)
        ]

        task_config.freeze()

        configs.append(task_config.clone())

    return configs


def distribute(
    data: List[str],
    scene_splits: List[List],
    num_gpus=8,
    procs_per_gpu=4,
    proc_offset=0,
    scenes_per_process=1,
) -> None:
    for idx, scene in enumerate(data):
        i = (idx // num_gpus) % scenes_per_process
        j = idx % num_gpus
        scene_splits[j * procs_per_gpu + i + proc_offset].append(scene)


def get_habitat_config(path: str):
    assert (
        path[-4:].lower() == ".yml" or path[-5:].lower() == ".yaml"
    ), f"path ({path}) must be a .yml or .yaml file."

    if not os.path.isabs(path):
        candidate_paths = [
            os.path.join(d, path)
            for d in [os.getcwd(), HABITAT_BASE, HABITAT_CONFIGS_DIR]
        ]
        success = False
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                success = True
                path = candidate_path
                break

        if not success:
            raise FileExistsError(
                f"Could not find config file with given relative path {path}. Tried the following possible absolute"
                f" paths {candidate_paths}."
            )
    elif not os.path.exists(path):
        raise FileExistsError(f"Could not find config file with given path {path}.")

    return habitat.get_config(path)


class BaseConfig(ExperimentConfig, ABC):
    """The base object navigation configuration file."""

    STEP_SIZE = 0.25
    ROTATION_DEGREES = 30.0
    VISIBILITY_DISTANCE = 1.0
    STOCHASTIC = True
    HORIZONTAL_FIELD_OF_VIEW = 90

    CAMERA_WIDTH = 224  # 672
    CAMERA_HEIGHT = 224  # 672
    SCREEN_SIZE = 224
    MAX_STEPS = 500

    ADVANCE_SCENE_ROLLOUT_PERIOD: Optional[int] = None
    SENSORS: Sequence[Sensor] = []

    def __init__(self):
        self.REWARDS_CONFIG = {
            "step_penalty": -0.01,
            "goal_success_reward": 10.0,
            "failed_stop_reward": 0.0,
            "shaping_weight": 1.0,
            "failed_action_penalty": -0.03,
            "visited_reward": 0.1,
            "object_reward": 0.4,
        }

    @classmethod
    def preprocessors(cls) -> Sequence[Union[Preprocessor, Builder[Preprocessor]]]:
        return tuple()


class ObjectNavHabitatBaseConfig(BaseConfig, ABC):
    """The base config for all AI2-THOR ObjectNav experiments."""

    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = tuple(
        range(torch.cuda.device_count())
    )  # (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = tuple(
        range(torch.cuda.device_count())
    )  # (torch.cuda.device_count() - 1,)
    NUM_PROCESSES = 40 if torch.cuda.is_available() else 1

    # DISTANCE_TO_GOAL = 1.0

    THOR_IS_HEADLESS: bool = False

    task_data_dir_template = os.path.join(
        HABITAT_DATASETS_DIR, "objectnav/mp3d/v1/{}/{}.json.gz"
    )
    TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
    VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
    TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

    # config = get_habitat_config(
    #     os.path.join(HABITAT_CONFIGS_DIR, "benchmark/nav/objectnav/objectnav_mp3d.yaml")
    # )
    config = get_habitat_config("configs/models/objectnav/objectnav_mp3d_v1_cow.yaml")

    with read_write(config):
        config.num_environments = 1
        # config.simulator_gpu_id = DEFAULT_TRAIN_GPU_IDS
        config.habitat.dataset.data_path = config.habitat.dataset.data_path.replace(
            "{split}", config.habitat.dataset.split
        )
        # ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]

        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = (
            False
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = (
            20.0
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = (
            0.05
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )

        config.habitat.simulator.turn_angle = int(BaseConfig.ROTATION_DEGREES)
        config.habitat.simulator.forward_step_size = BaseConfig.STEP_SIZE
        config.habitat.environment.max_episode_steps = BaseConfig.MAX_STEPS

        config.habitat.dataset.type = "ObjectNav-v1"
        # config.TASK.SUCCESS_DISTANCE = 1.0  # DISTANCE_TO_GOAL
        # config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"]
        # config.TASK.SPL.TYPE = "SPL"
        # config.TASK.SPL.SUCCESS_DISTANCE = 0.1  # DISTANCE_TO_GOAL
        # config.TASK.SUCCESS.SUCCESS_DISTANCE = 0.1  # DISTANCE_TO_GOAL

        config.mode = "train"

    TRAIN_CONFIGS = construct_env_configs(config)

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = True,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        randomize_train_materials: bool = False,
    ):
        super().__init__()

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(num_train_processes, self.NUM_PROCESSES)
        self.num_test_processes = v_or_default(
            num_test_processes, (10 if torch.cuda.is_available() else 1)
        )
        self.test_on_validation = test_on_validation
        self.train_gpu_ids = v_or_default(train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS)
        self.val_gpu_ids = v_or_default(val_gpu_ids, self.DEFAULT_VALID_GPU_IDS)
        self.test_gpu_ids = v_or_default(test_gpu_ids, self.DEFAULT_TEST_GPU_IDS)

        self.sampler_devices = self.train_gpu_ids
        self.randomize_train_materials = randomize_train_materials

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[torch.device] = []
        devices: Sequence[torch.device]
        if mode == "train":
            workers_per_device = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cast(Tuple, self.train_gpu_ids) * workers_per_device
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_train_processes, max(len(devices), 1)
            )
            sampler_devices = self.sampler_devices
        elif mode == "valid":
            nprocesses = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.val_gpu_ids
            )
        elif mode == "test":
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.test_gpu_ids
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_test_processes, max(len(devices), 1)
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices
            if mode == "train"
            else devices,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind].clone()
        config.defrost()

        habitat_sensors = []
        for s in self.SENSORS:
            for t in VISION_SENSOR_TO_HABITAT_LABEL:
                if isinstance(s, t):
                    habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                    break

        # if self.REWARDS_CONFIG["reward_type"] == "supervised":
        #     habitat_sensors.append("SEMANTIC_SENSOR")

        config.habitat.simulator.AGENT_0.SENSORS = habitat_sensors
        config.freeze()

        res = {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "distance_to_goal": self.DISTANCE_TO_GOAL,  # type:ignore
            "rewards_config": self.REWARDS_CONFIG,
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
        }
        res["loop_dataset"] = True
        res["allow_flipping"] = True
        res["randomize_materials_in_training"] = self.randomize_train_materials

        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.config.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"

        habitat_sensors = []
        for s in self.SENSORS:
            for t in VISION_SENSOR_TO_HABITAT_LABEL:
                if isinstance(s, t):
                    habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                    break

        config.simulator.AGENT_0.SENSORS = habitat_sensors
        config.freeze()

        res = {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "rewards_config": self.REWARDS_CONFIG,
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
        }
        res["loop_dataset"] = False

        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if self.test_on_validation or self.TEST_SCENES is None:
            if not self.test_on_validation:
                get_logger().warning(
                    "No test dataset dir detected, running test on validation set instead."
                )
            else:
                get_logger().info(
                    "`test_on_validation` was `True``, running test on validation set."
                )
            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )

        else:
            config = self.config.clone()
            config.defrost()
            config.DATASET.DATA_PATH = self.TEST_SCENES
            config.MODE = "test"

            habitat_sensors = []
            for s in self.SENSORS:
                for t in VISION_SENSOR_TO_HABITAT_LABEL:
                    if isinstance(s, t):
                        habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                        break

            config.simulator.AGENT_0.SENSORS = habitat_sensors
            config.freeze()

            res = {
                "env_config": config,
                "max_steps": self.MAX_STEPS,
                "sensors": self.SENSORS,
                "action_space": gym.spaces.Discrete(
                    len(ObjectNavTask.class_action_names())
                ),
                "distance_to_goal": self.DISTANCE_TO_GOAL,
                "rewards_config": self.REWARDS_CONFIG,
                "seed": seeds[process_ind] if seeds is not None else None,
                "deterministic_cudnn": deterministic_cudnn,
            }
            res["loop_dataset"] = False
            return res

class PINHabitatBaseConfig(BaseConfig, ABC):
    """The base config for all PIN experiments."""

    DEFAULT_TRAIN_GPU_IDS = tuple(range(torch.cuda.device_count()))
    DEFAULT_VALID_GPU_IDS = tuple(
        range(torch.cuda.device_count())
    )  # (torch.cuda.device_count() - 1,)
    DEFAULT_TEST_GPU_IDS = tuple(
        range(torch.cuda.device_count())
    )  # (torch.cuda.device_count() - 1,)
    NUM_PROCESSES = 40 if torch.cuda.is_available() else 1

    # DISTANCE_TO_GOAL = 1.0

    THOR_IS_HEADLESS: bool = False

    task_data_dir_template = os.path.join(
        HABITAT_DATASETS_DIR, "pin/hm3d/v2/{}/{}.json.gz"
    )
    TRAIN_SCENES = task_data_dir_template.format(*(["train"] * 2))
    VALID_SCENES = task_data_dir_template.format(*(["val"] * 2))
    TEST_SCENES = task_data_dir_template.format(*(["test"] * 2))

    # config = get_habitat_config(
    #     os.path.join(HABITAT_CONFIGS_DIR, "benchmark/nav/objectnav/objectnav_mp3d.yaml")
    # )
    config = get_habitat_config("configs/models/pin/pin_hm3d_3ref_v1_cow.yaml")

    with read_write(config):
        config.num_environments = 1
        # config.simulator_gpu_id = DEFAULT_TRAIN_GPU_IDS
        config.habitat.dataset.data_path = config.habitat.dataset.data_path.replace(
            "{split}", config.habitat.dataset.split
        )
        # ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]

        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth = (
            False
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth = (
            20.0
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth = (
            0.05
        )

        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.width = (
            BaseConfig.CAMERA_WIDTH
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.height = (
            BaseConfig.CAMERA_HEIGHT
        )
        config.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor.hfov = (
            BaseConfig.HORIZONTAL_FIELD_OF_VIEW
        )

        config.habitat.simulator.turn_angle = int(BaseConfig.ROTATION_DEGREES)
        config.habitat.simulator.forward_step_size = BaseConfig.STEP_SIZE
        config.habitat.environment.max_episode_steps = BaseConfig.MAX_STEPS

        config.habitat.dataset.type = "PIN-v2"
        # config.TASK.SUCCESS_DISTANCE = 1.0  # DISTANCE_TO_GOAL
        # config.TASK.MEASUREMENTS = ["DISTANCE_TO_GOAL", "SUCCESS", "SPL", "SOFT_SPL"]
        # config.TASK.SPL.TYPE = "SPL"
        # config.TASK.SPL.SUCCESS_DISTANCE = 0.1  # DISTANCE_TO_GOAL
        # config.TASK.SUCCESS.SUCCESS_DISTANCE = 0.1  # DISTANCE_TO_GOAL

        config.mode = "train"

    TRAIN_CONFIGS = construct_env_configs(config)

    def __init__(
        self,
        num_train_processes: Optional[int] = None,
        num_test_processes: Optional[int] = None,
        test_on_validation: bool = True,
        train_gpu_ids: Optional[Sequence[int]] = None,
        val_gpu_ids: Optional[Sequence[int]] = None,
        test_gpu_ids: Optional[Sequence[int]] = None,
        randomize_train_materials: bool = False,
    ):
        super().__init__()

        def v_or_default(v, default):
            return v if v is not None else default

        self.num_train_processes = v_or_default(num_train_processes, self.NUM_PROCESSES)
        self.num_test_processes = v_or_default(
            num_test_processes, (10 if torch.cuda.is_available() else 1)
        )
        self.test_on_validation = test_on_validation
        self.train_gpu_ids = v_or_default(train_gpu_ids, self.DEFAULT_TRAIN_GPU_IDS)
        self.val_gpu_ids = v_or_default(val_gpu_ids, self.DEFAULT_VALID_GPU_IDS)
        self.test_gpu_ids = v_or_default(test_gpu_ids, self.DEFAULT_TEST_GPU_IDS)

        self.sampler_devices = self.train_gpu_ids
        self.randomize_train_materials = randomize_train_materials

    def machine_params(self, mode="train", **kwargs):
        sampler_devices: Sequence[torch.device] = []
        devices: Sequence[torch.device]
        if mode == "train":
            workers_per_device = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else cast(Tuple, self.train_gpu_ids) * workers_per_device
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_train_processes, max(len(devices), 1)
            )
            sampler_devices = self.sampler_devices
        elif mode == "valid":
            nprocesses = 1
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.val_gpu_ids
            )
        elif mode == "test":
            devices = (
                [torch.device("cpu")]
                if not torch.cuda.is_available()
                else self.test_gpu_ids
            )
            nprocesses = evenly_distribute_count_into_bins(
                self.num_test_processes, max(len(devices), 1)
            )
        else:
            raise NotImplementedError("mode must be 'train', 'valid', or 'test'.")

        sensors = [*self.SENSORS]
        if mode != "train":
            sensors = [s for s in sensors if not isinstance(s, ExpertActionSensor)]

        sensor_preprocessor_graph = (
            SensorPreprocessorGraph(
                source_observation_spaces=SensorSuite(sensors).observation_spaces,
                preprocessors=self.preprocessors(),
            )
            if mode == "train"
            or (
                (isinstance(nprocesses, int) and nprocesses > 0)
                or (isinstance(nprocesses, Sequence) and sum(nprocesses) > 0)
            )
            else None
        )

        return MachineParams(
            nprocesses=nprocesses,
            devices=devices,
            sampler_devices=sampler_devices
            if mode == "train"
            else devices,  # ignored with > 1 gpu_ids
            sensor_preprocessor_graph=sensor_preprocessor_graph,
        )

    @classmethod
    def make_sampler_fn(cls, **kwargs) -> TaskSampler:
        return ObjectNavTaskSampler(**kwargs)

    def train_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.TRAIN_CONFIGS[process_ind].clone()
        config.defrost()

        habitat_sensors = []
        for s in self.SENSORS:
            for t in VISION_SENSOR_TO_HABITAT_LABEL:
                if isinstance(s, t):
                    habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                    break

        # if self.REWARDS_CONFIG["reward_type"] == "supervised":
        #     habitat_sensors.append("SEMANTIC_SENSOR")

        config.habitat.simulator.AGENT_0.SENSORS = habitat_sensors
        config.freeze()

        res = {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "distance_to_goal": self.DISTANCE_TO_GOAL,  # type:ignore
            "rewards_config": self.REWARDS_CONFIG,
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
        }
        res["loop_dataset"] = True
        res["allow_flipping"] = True
        res["randomize_materials_in_training"] = self.randomize_train_materials

        return res

    def valid_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        config = self.config.clone()
        config.defrost()
        config.DATASET.DATA_PATH = self.VALID_SCENES
        config.MODE = "validate"

        habitat_sensors = []
        for s in self.SENSORS:
            for t in VISION_SENSOR_TO_HABITAT_LABEL:
                if isinstance(s, t):
                    habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                    break

        config.simulator.AGENT_0.SENSORS = habitat_sensors
        config.freeze()

        res = {
            "env_config": config,
            "max_steps": self.MAX_STEPS,
            "sensors": self.SENSORS,
            "action_space": gym.spaces.Discrete(
                len(ObjectNavTask.class_action_names())
            ),
            "distance_to_goal": self.DISTANCE_TO_GOAL,
            "rewards_config": self.REWARDS_CONFIG,
            "seed": seeds[process_ind] if seeds is not None else None,
            "deterministic_cudnn": deterministic_cudnn,
        }
        res["loop_dataset"] = False

        return res

    def test_task_sampler_args(
        self,
        process_ind: int,
        total_processes: int,
        devices: Optional[List[int]] = None,
        seeds: Optional[List[int]] = None,
        deterministic_cudnn: bool = False,
    ) -> Dict[str, Any]:
        if self.test_on_validation or self.TEST_SCENES is None:
            if not self.test_on_validation:
                get_logger().warning(
                    "No test dataset dir detected, running test on validation set instead."
                )
            else:
                get_logger().info(
                    "`test_on_validation` was `True``, running test on validation set."
                )
            return self.valid_task_sampler_args(
                process_ind=process_ind,
                total_processes=total_processes,
                devices=devices,
                seeds=seeds,
                deterministic_cudnn=deterministic_cudnn,
            )

        else:
            config = self.config.clone()
            config.defrost()
            config.DATASET.DATA_PATH = self.TEST_SCENES
            config.MODE = "test"

            habitat_sensors = []
            for s in self.SENSORS:
                for t in VISION_SENSOR_TO_HABITAT_LABEL:
                    if isinstance(s, t):
                        habitat_sensors.append(VISION_SENSOR_TO_HABITAT_LABEL[t])
                        break

            config.simulator.AGENT_0.SENSORS = habitat_sensors
            config.freeze()

            res = {
                "env_config": config,
                "max_steps": self.MAX_STEPS,
                "sensors": self.SENSORS,
                "action_space": gym.spaces.Discrete(
                    len(ObjectNavTask.class_action_names())
                ),
                "distance_to_goal": self.DISTANCE_TO_GOAL,
                "rewards_config": self.REWARDS_CONFIG,
                "seed": seeds[process_ind] if seeds is not None else None,
                "deterministic_cudnn": deterministic_cudnn,
            }
            res["loop_dataset"] = False
            return res

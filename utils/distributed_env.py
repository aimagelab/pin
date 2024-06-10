from habitat.core.env import Env
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
if TYPE_CHECKING:
    from omegaconf import DictConfig
from habitat.core.dataset import Dataset, Episode
from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.registration import make_task
from gym import spaces
from habitat.config import read_write

class DistributedEnv(Env):
    def __init__(
        self,
        config: "DictConfig",
        num_jobs: int,
        job_index: int,
        dataset: Optional[Dataset[Episode]] = None
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
            
        self.original_num_episodes = self._dataset.num_episodes

        num_job_episodes = self._dataset.num_episodes // num_jobs
        first_index = job_index * num_job_episodes
        last_index = first_index + num_job_episodes
        if job_index == num_jobs - 1:
            last_index = self._dataset.num_episodes
        self._dataset.episodes = self._dataset.episodes[first_index:last_index]
        print("Number of episodes: ", self._dataset.num_episodes)
        
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

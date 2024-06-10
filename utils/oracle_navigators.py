import numpy as np
from typing import Optional, Union

from omegaconf import DictConfig
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from utils.new_top_down_map import *  # noqa


class GreedyShortestPathFollower(ShortestPathFollower):
    def get_next_action(self, goal_pos: np.array) -> Optional[Union[int, np.array]]:
        """Returns the next action along the shortest path."""
        self._build_follower()
        assert self._follower is not None
        next_action = self._follower.next_action_along(goal_pos)
        return self._get_return_value(next_action)


class ShortestPathFollowerAgentObjectnav(Agent):
    def __init__(self, sim, task_config: DictConfig):
        self.possible_actions = list(task_config.habitat.task.actions.keys())
        self._sim = sim
        self._oracle = GreedyShortestPathFollower(self._sim, 0.2)
        self._config = task_config
        
    def reset(self):
        pass

    def act(self, observations, env=None):
        distances = []
        current_position = self._sim.agents[0].state.position
        all_goals = [
            [vp.agent_state.position, i, j]
            for i, goal in enumerate(env.current_episode.goals)
            for j, vp in enumerate(goal.view_points)
        ]
        for vp, _, _ in all_goals:
            distances.append(
                self._sim.geodesic_distance(current_position, vp)
                + np.linalg.norm(current_position - vp)
            )
        indices_in_order_of_distance = np.argsort(distances)
        for idx in indices_in_order_of_distance:
            goal_pos, g, v = all_goals[idx]
            try:
                next_action = self._oracle.get_next_action(np.array(goal_pos))
                break
            except:
                continue
        else:
            next_action = [0]
        action = self.possible_actions[np.array(next_action).argmax()]
        return {"action": action}


class ShortestPathFollowerAgentMultiON(Agent):
    def __init__(self, sim, task_config: DictConfig):
        self.possible_actions = list(task_config.habitat.task.actions.keys())
        self._sim = sim
        self._oracle = GreedyShortestPathFollower(self._sim, 0.2)
        self._config = task_config
        self._distance_to = (
            self._config.habitat.task.measurements.distance_to_curr_goal.distance_to
        )

    def reset(self):
        pass

    def act(self, observations, env=None):
        curr_goal_idx = env.task.current_goal_index
        if self._distance_to == "POINT":
            goal_pos = env.current_episode.goals[curr_goal_idx].position
            try:
                next_action = self._oracle.get_next_action(np.array(goal_pos))
            except:
                next_action = [0]
                with open("logs/multion_episodes_wo_spf.txt", "a") as f:
                    f.write(
                        f"id:{env.current_episode.episode_id}, goal_idx:{curr_goal_idx}, curr_pos:{self._sim.get_agent_state().position}, goal_pos:{env.current_episode.goals[curr_goal_idx].position}, goal_cat:{env.current_episode.goals[curr_goal_idx].object_category}\n"
                    )
        elif self._distance_to == "VIEW_POINTS":
            distances = []
            current_position = self._sim.agents[0].state.position
            for vp in env.current_episode.goals[curr_goal_idx].viewpoints:
                distances.append(
                    self._sim.geodesic_distance(current_position, vp)
                    + np.linalg.norm(current_position - vp)
                )
            indices_in_order_of_distance = np.argsort(distances)
            for i in indices_in_order_of_distance:
                goal_pos = env.current_episode.goals[curr_goal_idx].viewpoints[i]
                try:
                    next_action = self._oracle.get_next_action(np.array(goal_pos))
                    break
                except:
                    continue
            else:
                next_action = [0]
                with open("logs/multion_episodes_wo_spf.txt", "a") as f:
                    f.write(
                        f"id:{env.current_episode.episode_id}, goal_idx:{curr_goal_idx}, curr_pos:{self._sim.get_agent_state().position}, goal_pos:{env.current_episode.goals[curr_goal_idx].position}, goal_cat:{env.current_episode.goals[curr_goal_idx].object_category}\n"
                    )
        action = self.possible_actions[np.array(next_action).argmax()]
        return {"action": action}


class ShortestPathFollowerAgentPIN(Agent):
    def __init__(self, sim, task_config: DictConfig):
        self.possible_actions = list(task_config.habitat.task.actions.keys())
        self._sim = sim
        self._config = task_config
        self._oracle = GreedyShortestPathFollower(
            self._sim, 0.2
        )
        self._distance_to = (
            self._config.habitat.task.measurements.distance_to_goal.distance_to
        )

    def reset(self):
        pass

    def act(self, observations, env=None):
        current_position = self._sim.get_agent_state().position
        if self._distance_to == "POINT":
            distances = []
            for goal in env.current_episode.goals:
                gp = goal.position
                distances.append(
                    self._sim.geodesic_distance(current_position, gp)
                    + np.linalg.norm(current_position - gp)
                )
            curr_goal_idx = np.argmin(distances)
            goal_pos = env.current_episode.goals[curr_goal_idx].position
            try:
                next_action = self._oracle.get_next_action(np.array(goal_pos))
            except:
                next_action = [0]
                print("SPF failed to find a path")

        elif self._distance_to == "VIEW_POINTS":
            raise NotImplementedError
        action = self.possible_actions[np.array(next_action).argmax()]
        return {"action": action}

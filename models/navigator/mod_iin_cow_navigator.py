from home_robot.agent.imagenav_agent.imagenav_agent import ImageNavAgent, IINAgentModule
from home_robot.agent.imagenav_agent.visualizer import NavVisualizer
from home_robot.agent.imagenav_agent.obs_preprocessor import ObsPreprocessor
from home_robot.core.interfaces import DiscreteNavigationAction, Observations
from home_robot.mapping.semantic.categorical_2d_semantic_map_module import (
    Categorical2DSemanticMapModule,
)
from home_robot.mapping.semantic.categorical_2d_semantic_map_state import (
    Categorical2DSemanticMapState,
)
from home_robot.navigation_planner.discrete_planner import DiscretePlanner
import home_robot.utils.pose as pu

from torch import Tensor
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple, List, Any, Dict
from omegaconf import DictConfig
import torch
import cv2
import skimage
import os
from torchvision.transforms.functional import resize
import torchvision.transforms as T
from torchvision.transforms.transforms import Resize
from torch import device, is_tensor
import torch.nn as nn
import matplotlib.pyplot as plt

from models.detector.clip_on_wheels.owl import PersOwl
from models.detector.clip_on_wheels.dino import DINOMatcher
from home_robot.agent.imagenav_agent.frontier_exploration import FrontierExplorationPolicy

from home_robot.utils.constants import (
    MAX_DEPTH_REPLACEMENT_VALUE,
    MIN_DEPTH_REPLACEMENT_VALUE,
)

from utils.obs_processor import ModIINObsPreprocessor

class CowNavVisualizer(NavVisualizer):
    def __init__(
        self,
        num_sem_categories: int,
        map_size_cm: int,
        map_resolution: int,
        print_images: bool,
        dump_location: str,
        exp_name: str,
    ) -> None:
        """
        Arguments:
            num_sem_categories: number of semantic segmentation categories
            map_size_cm: global map size (in centimeters)
            map_resolution: size of map bins (in centimeters)
            print_images: if True, save visualization as images
            coco_categories_legend: path to the legend image of coco categories
        """
        self.print_images = print_images
        self.default_vis_dir = f"{dump_location}/images/{exp_name}"
        if self.print_images:
            os.makedirs(self.default_vis_dir, exist_ok=True)

        self.num_sem_categories = num_sem_categories
        self.map_resolution = map_resolution
        self.map_shape = (
            map_size_cm // self.map_resolution,
            map_size_cm // self.map_resolution,
        )

        self.vis_dir = None
        self.image_vis = None
        self.visited_map_vis = None
        self.last_xy = None
        self.ind_frame_height = 450
        
    def visualize(
        self,
        obstacle_map: np.ndarray,
        goal_map: np.ndarray,
        closest_goal_map: Optional[np.ndarray],
        sensor_pose: np.ndarray,
        found_goal: bool,
        explored_map: np.ndarray,
        semantic_frame: np.ndarray,
        timestep: int,
        last_goal_image,
        last_td_map: Dict[str, Any] = None,
        last_collisions: Dict[str, Any] = None,
        semantic_map: Optional[np.ndarray] = None,
        visualize_goal: bool = True,
        metrics: Dict[str, Any] = None,
        been_close_map=None,
        blacklisted_targets_map=None,
        frontier_map: Optional[np.ndarray] = None,
        dilated_obstacle_map: Optional[np.ndarray] = None,
        instance_map: Optional[np.ndarray] = None,
        short_term_goal: Optional[np.ndarray] = None,
        goal_pose=None,
    ) -> None:
        """Visualize frame input and semantic map.

        Args:
            obstacle_map: (M, M) binary local obstacle map prediction
            goal_map: (M, M) binary array denoting goal location
            closest_goal_map: (M, M) binary array denoting closest goal
             location in the goal map in geodesic distance
            sensor_pose: (7,) array denoting global pose (x, y, o)
             and local map boundaries planning window (gy1, gy2, gx1, gy2)
            found_goal: whether we found the object goal category
            explored_map: (M, M) binary local explored map prediction
            semantic_map: (M, M) local semantic map predictions
            semantic_frame: semantic frame visualization
            timestep: time step within the episode
            last_td_map: habitat oracle top down map
            last_collisions: collisions dictionary
            visualize_goal: if True, visualize goal
            metrics: can populate for last frame
        """
        if not self.print_images:
            return

        if last_collisions is None:
            last_collisions = {"is_collision": False}

        if dilated_obstacle_map is not None:
            obstacle_map = dilated_obstacle_map

        goal_frame = self.make_goal(last_goal_image)
        obs_frame = self.make_observations(
            semantic_frame,
            last_collisions["is_collision"],
            found_goal,
            metrics,
        )
        map_pred_frame = self.make_map_preds(
            sensor_pose,
            obstacle_map,
            explored_map,
            semantic_map,
            closest_goal_map,
            goal_map,
            visualize_goal,
        )
        td_map_frame = None if last_td_map is None else self.make_td_map(last_td_map)

        kp_frame = np.ones_like(goal_frame) * 255
        kp_frame = self.make_attention(timestep)

        if td_map_frame is None:
            frame = np.concatenate(
                [goal_frame, obs_frame, map_pred_frame, kp_frame], axis=1
            )
        else:
            upper_frame = np.concatenate([goal_frame, obs_frame, kp_frame], axis=1)
            lower_frame = self.pad_frame(
                np.concatenate([map_pred_frame, td_map_frame], axis=1),
                upper_frame.shape[1],
            )
            frame = np.concatenate([upper_frame, lower_frame], axis=0)

        nframes = 1 if metrics is None else 5
        for i in range(nframes):
            name = f"snapshot_{timestep}_{i}.png"
            cv2.imwrite(os.path.join(self.vis_dir, name), frame)
                
    def make_attention(self, timestep: int) -> np.ndarray:
        """Create an attention frame."""
        fname = os.path.join(self.vis_dir, f"attention_{timestep}.png")
        assert os.path.exists(fname), f"attention frame does not exist at `{fname}`."
        
        border_size = 10
        text_bar_height = 50 - border_size
        kp_img = cv2.imread(fname)
        os.remove(fname)
        
        new_h = self.ind_frame_height - text_bar_height - 2 * border_size
        new_w = int((new_h / kp_img.shape[0]) * kp_img.shape[1])
        kp_img = cv2.resize(kp_img, (new_w, new_h))

        kp_img = self._add_border(kp_img, border_size)

        w = kp_img.shape[1]
        top_bar = np.ones((text_bar_height, w, 3), dtype=np.uint8) * 255
        frame = np.concatenate([top_bar, kp_img.astype(np.uint8)], axis=0)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        color = (20, 20, 20)
        thickness = 2

        text = "Matches"
        textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
        textX = (w - textsize[0]) // 2
        textY = (text_bar_height + border_size + textsize[1]) // 2
        frame = cv2.putText(
            frame,
            text,
            (textX, textY),
            font,
            fontScale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        return frame

class CowAgent(ImageNavAgent):
    def __init__(self, config: DictConfig, device_id: int = 0) -> None:
        self.device = torch.device(f"cuda:{device_id}")
        self.obs_preprocessor = CowObsPreprocessor(config, self.device)

        self.max_steps = config.habitat.environment.max_episode_steps
        self.num_environments = 1

        self._module = CoWAgentPINModule(config).to(self.device)

        self.use_dilation_for_stg = config.planner.use_dilation_for_stg
        self.verbose = config.planner.verbose

        self.semantic_map = Categorical2DSemanticMapState(
            device=self.device,
            num_environments=self.num_environments,
            num_sem_categories=config.semantic_map.num_sem_categories,
            map_resolution=config.semantic_map.map_resolution,
            map_size_cm=config.semantic_map.map_size_cm,
            global_downscaling=config.semantic_map.global_downscaling,
        )
        agent_radius_cm = config.habitat.simulator.agents.main_agent.radius * 100.0
        agent_cell_radius = int(
            np.ceil(agent_radius_cm / config.semantic_map.map_resolution)
        )
        self.planner = DiscretePlanner(
            turn_angle=config.habitat.simulator.turn_angle,
            collision_threshold=config.planner.collision_threshold,
            step_size=config.planner.step_size,
            obs_dilation_selem_radius=config.planner.obs_dilation_selem_radius,
            goal_dilation_selem_radius=config.planner.goal_dilation_selem_radius,
            map_size_cm=config.semantic_map.map_size_cm,
            map_resolution=config.semantic_map.map_resolution,
            visualize=False,
            print_images=False,
            dump_location=config.dump_location,
            exp_name=config.exp_name,
            agent_cell_radius=agent_cell_radius,
            min_obs_dilation_selem_radius=config.planner.min_obs_dilation_selem_radius,
            map_downsample_factor=config.planner.map_downsample_factor,
            map_update_frequency=config.planner.map_update_frequency,
            discrete_actions=config.planner.discrete_actions,
        )

        self.goal_filtering = config.semantic_prediction.goal_filtering
        self.goal_update_steps = self._module.goal_update_steps
        self.timesteps = None
        self.timesteps_before_goal_update = None
        self.found_goal = torch.zeros(
            self.num_environments, 1, dtype=bool, device=self.device
        )
        self.goal_map = torch.zeros(
            self.num_environments,
            1,
            *self.semantic_map.local_map.shape[2:],
            dtype=self.semantic_map.local_map.dtype,
            device=self.device,
        )

        self.visualizer = None
        if config.generate_videos:
            self.visualizer = CowNavVisualizer(
                num_sem_categories=config.semantic_map.num_sem_categories,
                map_size_cm=config.semantic_map.map_size_cm,
                map_resolution=config.semantic_map.map_resolution,
                print_images=config.generate_videos,
                dump_location=config.dump_location,
                exp_name=config.exp_name,
            )

    def act(self, obs, env):
        """Act end-to-end."""
        extra_dilation = 0
        captions = obs['task_observations']['pin_goal'][1] if 'pin_goal' in obs['task_observations'] and type(obs['task_observations']['pin_goal'][1]) == list else None
        while extra_dilation < 200:
            try:
                (
                    obs_preprocessed,
                    pose_delta,
                    camera_pose,
                    num_matches
                ) = self.obs_preprocessor.preprocess(obs, extra_dilation=extra_dilation, captions=captions)

                planner_inputs, vis_inputs, matched = self._prepare_planner_inputs(
                    obs_preprocessed, pose_delta, num_matches, camera_pose
                )

                closest_goal_map = None
                if self.timesteps[0] >= (self.max_steps - 1):
                    action = DiscreteNavigationAction.STOP
                else:
                    action, closest_goal_map, _, _ = self.planner.plan(
                        **planner_inputs[0],
                        use_dilation_for_stg=self.use_dilation_for_stg,
                        debug=self.verbose,
                    )
            except Exception as e:
                if type(e) == ValueError:
                    # raise e
                    print(e)
                    print("Probably goal has been detected but not projected. Increasing the dilation size.")
                    extra_dilation += 2
                    continue
                else:
                    raise e
            break
        if extra_dilation >= 200:
            print("Goal not detected. Stopping.")
            action = DiscreteNavigationAction.STOP
            matched = True

        if self.visualizer is not None:
            # collision = obs['task_observations'].get("collisions")
            collision = None
            if collision is None:
                collision = {"is_collision": False}
            min_size = min(obs['task_observations']["pin_goal"][0][0].shape[0:2])
            info = {
                **planner_inputs[0],
                **vis_inputs[0],
                "semantic_frame": obs['rgb'],
                "closest_goal_map": closest_goal_map,
                "last_goal_image": obs['task_observations']["pin_goal"][0][0][0:min_size, 0:min_size, :3],
                "last_collisions": collision,
                "last_td_map": None,
                # "num_refs": obs['task_observations']["pin_goal"].shape[0]
            }
            self.visualizer.visualize(**info)
            
        if action == DiscreteNavigationAction.STOP:
            action = {"action": "stop"}
        elif action == DiscreteNavigationAction.TURN_LEFT:
            action = {"action": "turn_left"}
        elif action == DiscreteNavigationAction.TURN_RIGHT:
            action = {"action": "turn_right"}
        elif action == DiscreteNavigationAction.MOVE_FORWARD:
            action = {"action": "move_forward"}
        else:
            raise ValueError(f"Invalid action: {action}")
        
        return action, matched
    
    @torch.no_grad()
    def _prepare_planner_inputs(
        self,
        obs: torch.Tensor,
        pose_delta: torch.Tensor,
        num_matches: int,
        camera_pose: Optional[torch.Tensor] = None,
    ) -> Tuple[List[dict], List[dict]]:
        """
        Determine a long-term navigation goal in 2D map space for a local policy to
        execute.
        """
        dones = torch.zeros(self.num_environments, dtype=torch.bool)
        update_global = torch.tensor(
            [
                self.timesteps_before_goal_update[e] == 0
                for e in range(self.num_environments)
            ]
        )

        (
            self.goal_map,
            self.found_goal,
            frontier_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            matched
        ) = self._module(
            obs.unsqueeze(1),
            pose_delta.unsqueeze(1),
            dones.unsqueeze(1),
            update_global.unsqueeze(1),
            camera_pose,
            self.found_goal,
            self.goal_map,
            self.semantic_map.local_map,
            self.semantic_map.global_map,
            self.semantic_map.local_pose,
            self.semantic_map.global_pose,
            self.semantic_map.lmb,
            self.semantic_map.origins,
            num_matches
        )

        self.semantic_map.local_pose = seq_local_pose[:, -1]
        self.semantic_map.global_pose = seq_global_pose[:, -1]
        self.semantic_map.lmb = seq_lmb[:, -1]
        self.semantic_map.origins = seq_origins[:, -1]

        goal_map = self._prep_goal_map_input()
        for e in range(self.num_environments):
            self.semantic_map.update_frontier_map(e, frontier_map[e][0].cpu().numpy())
            if self.found_goal[e].item():
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
            elif self.timesteps_before_goal_update[e] == 0:
                self.semantic_map.update_global_goal_for_env(e, goal_map[e])
                self.timesteps_before_goal_update[e] = self.goal_update_steps

        self.timesteps = [self.timesteps[e] + 1 for e in range(self.num_environments)]
        self.timesteps_before_goal_update = [
            self.timesteps_before_goal_update[e] - 1
            for e in range(self.num_environments)
        ]

        planner_inputs = [
            {
                "obstacle_map": self.semantic_map.get_obstacle_map(e),
                "goal_map": self.semantic_map.get_goal_map(e),
                "frontier_map": self.semantic_map.get_frontier_map(e),
                "sensor_pose": self.semantic_map.get_planner_pose_inputs(e),
                "found_goal": self.found_goal[e].item(),
            }
            for e in range(self.num_environments)
        ]
        vis_inputs = [
            {
                "explored_map": self.semantic_map.get_explored_map(e),
                "timestep": self.timesteps[e],
            }
            for e in range(self.num_environments)
        ]
        if self.semantic_map.num_sem_categories > 1:
            for e in range(self.num_environments):
                vis_inputs[e]["semantic_map"] = self.semantic_map.get_semantic_map(e)

        return planner_inputs, vis_inputs, matched


class CowObsPreprocessor(ModIINObsPreprocessor):
    def __init__(self, config: DictConfig, device: torch.device, open_clip_checkpoint: str = "", alpha: float = 0.0) -> None:
        self.device = device
        self.frame_height = config.frame_height
        self.frame_width = config.frame_width

        self.depth_filtering = config.semantic_prediction.depth_filtering
        self.depth_filter_range_cm = config.semantic_prediction.depth_filter_range_cm
        self.preprojection_kp_dilation = config.preprojection_kp_dilation
        # self.match_projection_threshold = config.superglue.match_projection_threshold

        # init episode variables
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = None
        self.step = None
        
        self.multiscale = config.multiscale
        self.scales = config.scales
        
        self.resize_dim = config.cow.resize_dim
        
        if config.cow.method == "dino":
            kwargs = {
                "model_name": config.cow.model_name,
                "match_threshold": config.cow.match_threshold,
                "device": device,
                "resize_dim": config.cow.resize_dim
            }
        elif config.cow.method == "clip_text":
            kwargs = {
                "model_name": config.cow.model_name,
                "model_weights": config.cow.model_weights,
                "match_threshold": config.cow.match_threshold,
                "device": device,
                "resize_dim": config.cow.resize_dim,
                "templates": config.cow.templates
            }
        else:
            kwargs = {
                "clip_model_name": config.cow.clip_model_name,
                "classes": config.cow.classes,
                "classes_clip": config.cow.classes_clip,
                "templates": config.cow.templates,
                "threshold": config.cow.threshold,
                "device": device,
                "center_only": config.cow.center_only,
                "modality": config.cow.modality
            }

        if config.cow.method == "owl":
            self.clip_module = PersOwl(**kwargs)
        elif config.cow.method == "dino":
            self.clip_module = DINOMatcher(**kwargs)
        else:
            raise NotImplementedError("Method not supported")
        
        if open_clip_checkpoint is not None and os.path.exists(open_clip_checkpoint):
            self.clip_module.load_weight_from_open_clip(open_clip_checkpoint, alpha)

        self.transform = T.Compose([T.ToPILImage()])
        self.default_vis_dir = f"{config.dump_location}/images/{config.exp_name}"
        self.print_images = config.generate_videos,
        
    def reset(self) -> None:
        """Reset for a new episode since pre-processing is temporally dependent."""
        self.goal_image = None
        self.goal_image_keypoints = None
        self.goal_mask = None
        self.last_pose = np.zeros(3)
        self.step = 0
        self.clip_module.reset()

    def make_attention_plot(self, attention):
        path = os.path.join(self.default_vis_dir, f"attention_{self.step+1}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        kernel = np.ones((4, 4), np.uint8) 
        plot_attention = cv2.dilate(attention.cpu().numpy().astype(np.uint8)*255, kernel=kernel, iterations=1)
        plt.imshow(plot_attention, cmap="hot", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(path), bbox_inches="tight", pad_inches=0)
        plt.close()
        
    def localize_object(self, observations, captions=None) -> Tuple[int, float]:
        img_tensor = None
        
        if observations["rgb"].shape[0] != observations["rgb"].shape[1]:
            for transform in self.transform.transforms:
                if type(transform) == Resize:
                    if type(transform.size) == int:
                        transform.size = (transform.size, transform.size)

        if self.transform is not None:
            img_tensor = self.transform(observations["rgb"])
        else:
            img_tensor = observations["rgb"]

        # will always be true but not for ViT-OWL localization
        if is_tensor(img_tensor):
            img_tensor = img_tensor.unsqueeze(0)
            
        # if type(self.clip_module) == PerSAMMatcher:
        #     result = self.clip_module(img_tensor, observations['task_observations']["pin_goal"])
        #     if self.print_images:
        #         self.make_attention_plot(result)
        #     return result
            
        # else:

        # NOTE: child must set clip_module
        if "object_goal" in observations['task_observations']:
            return self.clip_module(img_tensor, observations["object_goal"])
        elif "pin_goal" in observations['task_observations']:
            # img_tensor1 = img_tensor[]
            if img_tensor.size[0] < img_tensor.size[1]:
                img_tensor1 = img_tensor.crop((0, 0, img_tensor.size[0], img_tensor.size[0]))
                img_tensor1 = img_tensor1.resize((self.resize_dim, self.resize_dim))
                img_tensor2 = img_tensor.crop((0, img_tensor.size[1]-img_tensor.size[0], img_tensor.size[0], img_tensor.size[1]))
                img_tensor2 = img_tensor2.resize((self.resize_dim, self.resize_dim))
                result1 = self.clip_module(img_tensor1, observations['task_observations']["pin_goal"][0], category=captions)
                result1 = resize(result1.unsqueeze(0).unsqueeze(0), (img_tensor.size[0], img_tensor.size[0]), interpolation=T.InterpolationMode.NEAREST)[0, 0]
                result2 = self.clip_module(img_tensor2, observations['task_observations']["pin_goal"][0], category=captions)
                result2 = resize(result2.unsqueeze(0).unsqueeze(0), (img_tensor.size[0], img_tensor.size[0]), interpolation=T.InterpolationMode.NEAREST)[0, 0]
                result = torch.zeros((img_tensor.size[1], img_tensor.size[0]))
                result[0:img_tensor.size[0], 0:img_tensor.size[0]] += result1
                result[img_tensor.size[1]-img_tensor.size[0]:img_tensor.size[1], 0:img_tensor.size[0]] += result2
                result = torch.clamp(result, max=1.0)
                if self.print_images:
                    self.make_attention_plot(result)
                return result
            elif img_tensor.size[0] > img_tensor.size[1]:
                img_tensor1 = img_tensor.crop((0, 0, img_tensor.size[1], img_tensor.size[1]))
                img_tensor1 = img_tensor1.resize((self.resize_dim, self.resize_dim))
                img_tensor2 = img_tensor.crop((img_tensor.size[0]-img_tensor.size[1], 0, img_tensor.size[0], img_tensor.size[1]))
                img_tensor2 = img_tensor2.resize((self.resize_dim, self.resize_dim))
                result1 = self.clip_module(img_tensor1, observations['task_observations']["pin_goal"][0], category=None)
                result1 = resize(result1.unsqueeze(0).unsqueeze(0), (img_tensor.size[1], img_tensor.size[1]), interpolation=T.InterpolationMode.NEAREST)[0, 0]
                result2 = self.clip_module(img_tensor2, observations['task_observations']["pin_goal"][0], category=None)
                result2 = resize(result2.unsqueeze(0).unsqueeze(0), (img_tensor.size[1], img_tensor.size[1]), interpolation=T.InterpolationMode.NEAREST)[0, 0]
                result = torch.zeros((img_tensor.size[1], img_tensor.size[0]))
                result[0:img_tensor.size[1], 0:img_tensor.size[1]] += result1
                result[0:img_tensor.size[1], img_tensor.size[0]-img_tensor.size[1]:img_tensor.size[0]] += result2
                result = torch.clamp(result, max=1.0)
                if self.print_images:
                    self.make_attention_plot(result)
                return result
            else:
                return self.clip_module(img_tensor, observations['task_observations']["pin_goal"][0], category=None)
        else:
            raise NotImplementedError("No correct goal found in observations")

    def preprocess(
        self, obs: Observations, extra_dilation: int = 0, captions=None
    ) -> Tuple[Tensor, Optional[Tensor], ndarray, ndarray]:
        """
        Preprocess observations of a single timestep batched across
        environments.

        Arguments:
            obs: list of observations of length num_environments

        Returns:
            obs_preprocessed: frame containing (RGB, depth, keypoint_loc) of
               shape (3 + 1 + 1, frame_height, frame_width)
            pose_delta: sensor pose delta (dy, dx, dtheta) since last frame
               of shape (num_environments, 3)
            matches: keypoint correspondences from goal image to egocentric
               image of shape (1, n)
            confidence: keypoint correspondence confidence of shape (1, n)
            camera_pose: camera extrinsic pose of shape (num_environments, 4, 4)
        """

        pose_delta, self.last_pose = self._preprocess_pose_and_delta(obs)
        obs_preprocessed, num_matches = self._preprocess_frame(obs, extra_dilation=extra_dilation, captions=captions)

        # camera_pose = obs["camera_pose"]
        # if camera_pose is not None:
        #     camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        camera_pose = None

        self.step += 1
        return obs_preprocessed, pose_delta, camera_pose, num_matches

    def _preprocess_frame(self, obs: Observations, extra_dilation=0, captions=None) -> Tuple[Tensor, ndarray, ndarray]:
        """Preprocess frame information in the observation."""

        def downscale(rgb: ndarray, depth: ndarray) -> Tuple[ndarray, ndarray]:
            """downscale RGB and depth frames to self.frame_{width,height}"""
            ds = rgb.shape[1] / self.frame_width
            if ds == 1:
                return rgb, depth
            dim = (self.frame_width, self.frame_height)
            rgb = cv2.resize(rgb, dim, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)[:, :, None]
            return rgb, depth
        
        kp_loc = self.localize_object(obs, captions=captions).unsqueeze(-1)
        kp_loc[obs["depth"] == MIN_DEPTH_REPLACEMENT_VALUE] = 0.0
        kp_loc[obs["depth"] == MAX_DEPTH_REPLACEMENT_VALUE] = 0.0
        num_matches = kp_loc.sum()
        kp_loc = kp_loc.cpu().numpy()
        if self.preprojection_kp_dilation > 0:
            disk = skimage.morphology.disk(self.preprojection_kp_dilation + extra_dilation)
            kp_loc = np.expand_dims(cv2.dilate(kp_loc, disk, iterations=1), axis=2)

        depth = np.expand_dims(obs["depth"], axis=2) * 100.0
        rgb, depth = downscale(obs["rgb"], depth)
        
        obs_preprocessed = np.concatenate([rgb, depth, kp_loc], axis=2)
        obs_preprocessed = obs_preprocessed.transpose(2, 0, 1)
        obs_preprocessed = torch.from_numpy(obs_preprocessed)
        obs_preprocessed = obs_preprocessed.to(device=self.device)
        obs_preprocessed = obs_preprocessed.unsqueeze(0)
        return obs_preprocessed, num_matches

    def _preprocess_pose_and_delta(self, obs: Observations) -> Tuple[Tensor, ndarray]:
        """merge GPS+compass. Compute the delta from the previous timestep."""
        curr_pose = np.array([obs["gps"][0], obs["gps"][1], obs["compass"][0]])
        pose_delta = (
            torch.tensor(pu.get_rel_pose_change(curr_pose, self.last_pose))
            .unsqueeze(0)
            .to(device=self.device)
        )
        return pose_delta, curr_pose

class CoWAgentPINModule(IINAgentModule):
    def __init__(self, config: DictConfig) -> None:
        nn.Module.__init__(self)

        self.semantic_map_module = Categorical2DSemanticMapModule(
            frame_height=config.frame_height,
            frame_width=config.frame_width,
            camera_height=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[
                1
            ],
            hfov=config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov,
            num_sem_categories=config.semantic_map.num_sem_categories,
            map_size_cm=config.semantic_map.map_size_cm,
            map_resolution=config.semantic_map.map_resolution,
            vision_range=config.semantic_map.vision_range,
            explored_radius=config.semantic_map.explored_radius,
            been_close_to_radius=config.semantic_map.been_close_to_radius,
            global_downscaling=config.semantic_map.global_downscaling,
            du_scale=config.semantic_map.du_scale,
            cat_pred_threshold=config.semantic_map.cat_pred_threshold,
            exp_pred_threshold=config.semantic_map.exp_pred_threshold,
            map_pred_threshold=config.semantic_map.map_pred_threshold,
            must_explore_close=config.semantic_map.must_explore_close,
            min_obs_height_cm=config.semantic_map.min_obs_height_cm,
            dilate_obstacles=config.semantic_map.dilate_obstacles,
            dilate_size=config.semantic_map.dilate_size,
            dilate_iter=config.semantic_map.dilate_iter,
        )
        self.goal_policy_config = config.cow
        self.exploration_policy = FrontierExplorationPolicy()

    def check_goal_detection(
        self,
        goal_map: torch.Tensor,
        found_goal: torch.Tensor,
        local_map: torch.Tensor,
        num_matches: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Goal detection and localization"""
        
        matched = False

        for e in range(found_goal.shape[0]):
            # if the goal category is empty, the goal can't be found
            # if num_matches <= self.goal_policy_config.min_matches:
            #     continue
            if num_matches <= 0.0:
                continue

            found_goal[e] = True
            # Set goal_map to the last channel of the local semantic map
            goal_map[e, 0] = local_map[e, -1]
            matched = True

        return goal_map, found_goal, matched

    def forward(
        self,
        seq_obs: torch.Tensor,
        seq_pose_delta: torch.Tensor,
        seq_dones: torch.Tensor,
        seq_update_global: torch.Tensor,
        seq_camera_poses: Optional[torch.Tensor],
        seq_found_goal: torch.Tensor,
        seq_goal_map: torch.Tensor,
        init_local_map: torch.Tensor,
        init_global_map: torch.Tensor,
        init_local_pose: torch.Tensor,
        init_global_pose: torch.Tensor,
        init_lmb: torch.Tensor,
        init_origins: torch.Tensor,
        num_matches: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Reset the last channel of the local map each step when found_goal=False
        # init_local_map: [8, 21, 480, 480]
        init_local_map[:, -1][seq_found_goal[:, 0] == 0] *= 0.0

        # Update map with observations and generate map features
        batch_size, sequence_length = seq_obs.shape[:2]
        (
            seq_map_features,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
        ) = self.semantic_map_module(
            seq_obs,
            seq_pose_delta,
            seq_dones,
            seq_update_global,
            seq_camera_poses,
            init_local_map,
            init_global_map,
            init_local_pose,
            init_global_pose,
            init_lmb,
            init_origins,
        )

        # Predict high-level goals from map features.
        map_features = seq_map_features.flatten(0, 1)

        # the last channel of map_features is cut off -- used for goal det/loc.
        frontier_map = self.exploration_policy(map_features[:, :-1])
        seq_goal_map[seq_found_goal[:, 0] == 0] = frontier_map[
            seq_found_goal[:, 0] == 0
        ]

        # predict if the goal is found and where it is.
        seq_goal_map, seq_found_goal, matched = self.check_goal_detection(
            seq_goal_map, seq_found_goal, final_local_map, num_matches
        )
        seq_goal_map = seq_goal_map.view(
            batch_size, sequence_length, *seq_goal_map.shape[-2:]
        )

        seq_frontier_map = frontier_map.view(
            batch_size, sequence_length, *frontier_map.shape[-2:]
        )

        return (
            seq_goal_map,
            seq_found_goal,
            seq_frontier_map,
            final_local_map,
            final_global_map,
            seq_local_pose,
            seq_global_pose,
            seq_lmb,
            seq_origins,
            matched
        )

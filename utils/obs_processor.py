from home_robot.agent.imagenav_agent.obs_preprocessor import ObsPreprocessor
from home_robot.core.interfaces import Observations
import home_robot.utils.pose as pu

from torch import Tensor
import numpy as np
from numpy import ndarray
from typing import Optional, Tuple
from omegaconf import DictConfig
import torch
import cv2
import skimage
import torchvision
from torchvision.transforms.functional import resize

class ModIINObsPreprocessor(ObsPreprocessor):
    def __init__(self, config: DictConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self.multiscale = config.multiscale
        self.scales = config.scales
        
    @torch.no_grad()
    def squared_crop(self, images, masks, resize_dim=224, padding=10, device="cuda", avoid_resizing=False):
        # images = images.permute(0, 2, 3, 1).to(self.device)
        # masks = masks.to(self.device)
        shortest_side = int(min(images.shape[1:3]) / 2)
        image_size_x = images.shape[2]
        image_size_y = images.shape[1]
        batch_size = images.shape[0]
            
        # Find mask bounding boxes
        
        cumsum_x = masks.sum(dim=1).cumsum(dim=1).float()
        xmaxs = cumsum_x.argmax(dim=1, keepdim=True)
        cumsum_x[cumsum_x == 0] = np.inf
        xmins = cumsum_x.argmin(dim=1, keepdim=True)
        
        cumsum_y = masks.sum(dim=2).cumsum(dim=1).float()
        ymaxs = cumsum_y.argmax(dim=1, keepdim=True)
        cumsum_y[cumsum_y == 0] = np.inf
        ymins = cumsum_y.argmin(dim=1, keepdim=True)
        
        # Compute mask centers        
        mask_center_x = (xmaxs+xmins) / 2
        mask_center_y = (ymaxs+ymins) / 2
        
        # Get squared bounding boxes
        
        left_distance = (mask_center_x - xmins).unsqueeze(-1)
        right_distance = (xmaxs - mask_center_x).unsqueeze(-1)
        top_distance = (mask_center_y - ymins).unsqueeze(-1)
        bottom_distance = (ymaxs - mask_center_y).unsqueeze(-1)

        max_distance = torch.cat((left_distance, right_distance, top_distance, bottom_distance), dim=2).max(dim=2).values.int()
        max_distance[max_distance > shortest_side] = shortest_side
        
        del left_distance, right_distance, top_distance, bottom_distance
        
        xmaxs = mask_center_x + max_distance
        xmins = mask_center_x - max_distance
        ymaxs = mask_center_y + max_distance
        ymins = mask_center_y - max_distance
        
        xmins[xmaxs > image_size_x] = xmins[xmaxs > image_size_x] - (xmaxs[xmaxs > image_size_x] - image_size_x).int()
        xmaxs[xmaxs > image_size_x] = image_size_x
        ymins[ymaxs > image_size_y] = ymins[ymaxs > image_size_y] - (ymaxs[ymaxs > image_size_y] - image_size_y).int()
        ymaxs[ymaxs > image_size_y] = image_size_y
        xmaxs[xmins < 0] = xmaxs[xmins < 0] - xmins[xmins < 0]
        xmins[xmins < 0] = 0
        ymaxs[ymins < 0] = ymaxs[ymins < 0] - ymins[ymins < 0]
        ymins[ymins < 0] = 0

        batch_index = torch.arange(batch_size).unsqueeze(1).to(device)
        boxes = torch.cat((batch_index, xmins, ymins, xmaxs, ymaxs), 1)
        
        if avoid_resizing:
            resize_dim = int(xmaxs.max().item() - xmins.min().item())
        
        del xmins, ymins, xmaxs, ymaxs, mask_center_x, mask_center_y, cumsum_x, cumsum_y
                
        cropped_images = torchvision.ops.roi_align(images.permute(0, 3, 1, 2).float(), boxes.float(), resize_dim, aligned=True)
        cropped_masks = torchvision.ops.roi_align(masks.float().unsqueeze(1), boxes.float(), resize_dim, aligned=True).bool().squeeze(1)
        
        new_cropped_images = torch.ones((batch_size, 3, (resize_dim + padding*2), (resize_dim + padding*2)), device=device).int() * 255
        new_cropped_images[:, :, padding:resize_dim+padding, padding:resize_dim+padding] = cropped_images.int()
        new_cropped_images = resize(new_cropped_images.float(), (resize_dim, resize_dim)).int()
        
        new_cropped_masks = torch.zeros((batch_size, resize_dim + padding*2, resize_dim + padding*2), device=device).bool()
        new_cropped_masks[:, padding:resize_dim+padding, padding:resize_dim+padding] = cropped_masks.bool()
        new_cropped_masks = resize(new_cropped_masks.float(), (resize_dim, resize_dim)).bool()
        
        return new_cropped_images, new_cropped_masks

    def preprocess(
        self, obs: Observations
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
        if self.goal_image is None:
            img_goal = obs["task_observations"]["pin_goal"][:,:,:,0:3]
            self.goal_masks = [torch.from_numpy(reference[:,:,-1] / 255).bool() for reference in obs["task_observations"]["pin_goal"]]
            self.reference_occupancies = [goal_mask.sum() / torch.numel(goal_mask) for goal_mask in self.goal_masks]
            self.goal_image = []
            self.goal_image_keypoints = []
            if self.multiscale:
                paddings = self.scales
            else:
                paddings = [self.scales[0]]
            l = len(img_goal)
            new_goal_masks = []
            new_reference_occupancies = []
            for j, reference_img in enumerate(img_goal):
                for p, padding in enumerate(paddings):
                    mask = self.goal_masks[j].unsqueeze(0).to(self.device).bool()
                    image_tensor = torch.from_numpy(reference_img).unsqueeze(0).to(self.device)
                    image_tensor[~mask] = 255
                    image, mask = self.squared_crop(image_tensor, mask, avoid_resizing=False, padding=padding, resize_dim=min(obs['rgb'].shape[0:2]))
                    image = image.cpu()[0].permute(1,2,0).numpy().astype(np.uint8)
                    current_goal_image, current_goal_image_keypoints = self.matching.get_goal_image_keypoints(image)
                    # self.goal_masks[j*l + p] = mask.cpu()[0].numpy()
                    new_goal_masks.append(mask.cpu()[0].numpy())
                    new_reference_occupancies.append(self.reference_occupancies[j])
                    self.goal_image.append(current_goal_image)
                    self.goal_image_keypoints.append(current_goal_image_keypoints)
            
            self.goal_masks = new_goal_masks
            self.reference_occupancies = new_reference_occupancies

        pose_delta, self.last_pose = self._preprocess_pose_and_delta(obs)
        obs_preprocessed, matches, confidence, reference_occupancy, best_confidence_index = self._preprocess_frame(obs)

        # camera_pose = obs["camera_pose"]
        # if camera_pose is not None:
        #     camera_pose = torch.tensor(np.asarray(camera_pose)).unsqueeze(0)
        camera_pose = None

        self.step += 1
        return obs_preprocessed, pose_delta, camera_pose, matches, confidence, reference_occupancy, best_confidence_index

    def _preprocess_frame(self, obs: Observations) -> Tuple[Tensor, ndarray, ndarray]:
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

        def preprocess_keypoint_localization(
            rgb: ndarray,
            goal_keypoints: torch.Tensor,
            rgb_keypoints: torch.Tensor,
            matches: ndarray,
            confidence: ndarray,
            goal_mask: ndarray
        ) -> ndarray:
            """
            Given keypoint correspondences, determine the egocentric pixel coordinates
            of matched keypoints that lie within a mask of the goal object.
            """
            # map the valid goal keypoints to ego keypoints
            is_in_mask = goal_mask[goal_keypoints[:, 1], goal_keypoints[:, 0]]
            has_high_confidence = confidence >= self.match_projection_threshold
            is_matching_kp = matches > -1
            valid = np.logical_and(is_in_mask, has_high_confidence, is_matching_kp)
            # valid = np.logical_and(has_high_confidence, is_matching_kp)
            matched_rgb_keypoints = rgb_keypoints[matches[valid]]

            # set matched rgb keypoints as goal points
            kp_loc = np.zeros((*rgb.shape[:2], 1), dtype=rgb.dtype)
            kp_loc[matched_rgb_keypoints[:, 1], matched_rgb_keypoints[:, 0]] = 1

            if self.preprojection_kp_dilation > 0:
                disk = skimage.morphology.disk(self.preprojection_kp_dilation)
                kp_loc = np.expand_dims(cv2.dilate(kp_loc, disk, iterations=1), axis=2)

            return kp_loc

        depth = np.expand_dims(obs["depth"], axis=2) * 100.0
        rgb, depth = downscale(obs["rgb"], depth)

        goal_keypoints_list = []
        _, rgb_keypoints = self.matching.get_goal_image_keypoints(rgb)
        rgb_keypoints['keypoints1'] = rgb_keypoints['keypoints0']
        rgb_keypoints.pop('keypoints0')
        rgb_keypoints['scores1'] = rgb_keypoints['scores0']
        rgb_keypoints.pop('scores0')
        rgb_keypoints['descriptors1'] = rgb_keypoints['descriptors0']
        rgb_keypoints.pop('descriptors0')
        rgb_keypoints_list = []
        matches_list = []
        confidence_list = []
        for i, goal_image in enumerate(self.goal_image):
            (goal_keypoints_i, rgb_keypoints_i, matches_i, confidence_i) = self.matching(
                rgb,
                goal_image=goal_image,
                goal_image_keypoints=self.goal_image_keypoints[i],
                rgb_image_keypoints=rgb_keypoints,
                step=self.step,
                ref_index=i
            )
            goal_keypoints_list.append(goal_keypoints_i)
            rgb_keypoints_list.append(rgb_keypoints_i)
            matches_list.append(matches_i)
            confidence_list.append(confidence_i)
        
        best_confidence_score = -1
        best_confidence_index = 0
        for i, (goal_keypoints_i, rgb_keypoints_i, matches_i, confidence_i) in enumerate(zip(
            goal_keypoints_list, rgb_keypoints_list, matches_list, confidence_list)
        ):
            goal_keypoints_i = goal_keypoints_i[0].cpu().to(dtype=int).numpy()
            rgb_keypoints_i = rgb_keypoints_i[0].cpu().to(dtype=int).numpy()
            confidence_i = confidence_i[0]
            matches_i = matches_i[0]
            is_in_mask = self.goal_masks[i][goal_keypoints_i[:, 1], goal_keypoints_i[:, 0]]
            masked_confidence_i = confidence_i[is_in_mask]
            masked_matches_i = matches_i[is_in_mask]
            if masked_confidence_i.sum() > best_confidence_score:
                goal_keypoints = goal_keypoints_i
                rgb_keypoints = rgb_keypoints_i
                matches = matches_i
                confidence = confidence_i
                masked_confidence = masked_confidence_i
                masked_matches = masked_matches_i
                best_confidence_score = confidence_i.sum()
                reference_occupancy = self.reference_occupancies[i]
                goal_mask = self.goal_masks[i]
                best_confidence_index = i
        
        kp_loc = preprocess_keypoint_localization(
            rgb, goal_keypoints, rgb_keypoints, matches, confidence, goal_mask
        )

        obs_preprocessed = np.concatenate([rgb, depth, kp_loc], axis=2)
        obs_preprocessed = obs_preprocessed.transpose(2, 0, 1)
        obs_preprocessed = torch.from_numpy(obs_preprocessed)
        obs_preprocessed = obs_preprocessed.to(device=self.device)
        obs_preprocessed = obs_preprocessed.unsqueeze(0)
        return obs_preprocessed, masked_matches, masked_confidence, reference_occupancy, best_confidence_index

    def _preprocess_pose_and_delta(self, obs: Observations) -> Tuple[Tensor, ndarray]:
        """merge GPS+compass. Compute the delta from the previous timestep."""
        curr_pose = np.array([obs["gps"][0], obs["gps"][1], obs["compass"][0]])
        pose_delta = (
            torch.tensor(pu.get_rel_pose_change(curr_pose, self.last_pose))
            .unsqueeze(0)
            .to(device=self.device)
        )
        return pose_delta, curr_pose

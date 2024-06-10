import torch
import numpy as np
import timm
from torchvision.transforms import Compose, Resize, PILToTensor
from torchvision.transforms.functional import pil_to_tensor
from torch.nn.functional import interpolate
import torchvision
from math import sqrt
from PIL.Image import Image

class DINOMatcher(torch.nn.Module):
    def __init__(self,
                 model_name,
                 device="cuda",
                 max_batch_size=16,
                 resize_dim=518,
                 match_threshold=0.5):
        super(DINOMatcher, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            img_size=resize_dim,
        ).to(device).eval()
        data_config = timm.data.resolve_model_data_config(model_name)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.transform = Compose([
            Resize((resize_dim, resize_dim), antialias=None),
            lambda x: x / 255,
            self.transform.transforms[-1]
        ])
        self.device = device
        self.max_batch_size = max_batch_size
        self.match_threshold = match_threshold
        self.ref_features = None
        
    @torch.no_grad()
    def reset(self):
        self.ref_features = None
        
    @torch.no_grad()
    def extract_features(self, img):
        img = self.transform(img)
        features = self.model.forward_features(img.to(self.device).unsqueeze(0))
        num_tokens_side = int(sqrt(features.shape[1] - 1))
        return features[:, 1::, :].reshape(features.shape[0], num_tokens_side, num_tokens_side, features.shape[-1])
    
    @torch.no_grad()
    def batched_extract_features(self, images):
        if type(images) == np.ndarray:
            images = torch.from_numpy(images)
        batch_size = images.shape[0]
        if images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)
        images = self.transform(images).to(self.device)
        num_tokens = self.model.patch_embed.grid_size[0] * self.model.patch_embed.grid_size[1] + 1
        features = torch.zeros(batch_size, num_tokens, self.model.embed_dim, device=self.device)
        for i in range(0, batch_size, self.max_batch_size):
            if i + self.max_batch_size > batch_size:
                features[i:] = self.model.forward_features(images[i:])
            else:
                features[i:i + self.max_batch_size] = self.model.forward_features(images[i:i + self.max_batch_size])
        num_tokens_side = int(sqrt(features.shape[1] - 1))
        return features[:, 1::, :].reshape(batch_size, num_tokens_side, num_tokens_side, features.shape[-1])
    
    @torch.no_grad()
    def region_pooling(self, image_features, seg_masks):
        if type(seg_masks) == np.ndarray:
            seg_masks = torch.from_numpy(seg_masks)
        b, h, w, c = image_features.shape
        seg_masks = seg_masks.unsqueeze(1).float().to(image_features.device)
        seg_masks = interpolate(seg_masks, size=(h, w), mode='bilinear', align_corners=False)
        image_features = image_features.permute(0, 3, 1, 2)
        return ((image_features * seg_masks).sum(dim=(2, 3)) / (seg_masks.sum(dim=(2, 3)) + 1e-6)).unsqueeze(0)
    
    @torch.no_grad()
    def compute_similarity(self, references, obs):
        k, h_p, w_p, c = obs.shape
        obs = obs / obs.norm(dim=-1, keepdim=True) + 1e-6
        references = references / references.norm(dim=-1, keepdim=True) + 1e-6
        obs = obs.reshape(k, h_p * w_p, c)
        similarity = torch.einsum("bnc, kpc -> bnkp", references, obs)
        similarity = similarity.max(dim=1).values
        return similarity
    
    @torch.no_grad()
    def project_matches(self, matches, obs_height, obs_width):
        result = torch.zeros(obs_height, obs_width, device=matches.device)
        if matches.sum() == 0.0:
            return result
        matches_height, matches_width = matches.shape
        height_scale = obs_height / matches_height
        width_scale = obs_width / matches_width
        y = torch.arange(matches_height, device=matches.device).unsqueeze(1).expand(matches_height, matches_width)
        x = torch.arange(matches_width, device=matches.device).unsqueeze(0).expand(matches_height, matches_width)
        y = y[matches]
        x = x[matches]
        y = torch.round((y + 0.5) * height_scale).int()
        x = torch.round((x + 0.5) * width_scale).int()
        coordinates = torch.stack([y, x], dim=-1)
        for y, x in coordinates:
            result[y, x] = 1
        return result
        

    @torch.no_grad()
    def forward(self, obs, references, category=None):
        if type(obs) == Image:
            obs = pil_to_tensor(obs)
        obs_channels, obs_height, obs_width = obs.shape
        obs_features = self.extract_features(obs)
        if self.ref_features is None:
            self.ref_features = self.batched_extract_features(references[:,:,:,0:3])
            self.ref_features = self.region_pooling(self.ref_features, references[:,:,:,3])
        similarity = self.compute_similarity(self.ref_features, obs_features)
        # print(similarity.max().item())
        matches = similarity > self.match_threshold
        matches = matches.reshape(obs_features.shape[1], obs_features.shape[2])
        return self.project_matches(matches, obs_height, obs_width).cpu()
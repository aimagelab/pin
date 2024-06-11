from torchvision.transforms.functional import to_pil_image
import torch
import numpy as np
from torch import device
from typing import List

from transformers.models.owlvit.modeling_owlvit import OwlViTObjectDetectionOutput, OwlViTOutput

from utils.cow_utils.src.models.localization.clip_owl import ClipOwl, post_process
from models.detector.clip_on_wheels.utils import squared_crop

class PersOwl(ClipOwl):
    def __init__(
        self,
        clip_model_name: str,
        classes: List[str],
        classes_clip: List[str],
        templates: List[str],
        threshold: float,
        device: device,
        center_only: bool = False,
        modality: str = "category"
    ):
        super(PersOwl, self).__init__(
            clip_model_name,
            classes,
            classes_clip,
            templates,
            threshold,
            device,
            center_only,
        )
        self.modality = modality
        if self.modality == "text-to-image":
            self.blip_model = BlipModel(device=device)
        self.captions = None
        self.text_ids = [None, None, None]
        
    def forward(self, obs, references, category=None):
        # Args:
        # x: an observation image in PIL
        # o: a set of reference images as torch tensors
        # differently from ClipOwl, o is a set of reference images
        if self.modality == "category":
            return super(PersOwl, self).forward(obs, category, text_id=None)[0]
        elif self.modality == "captions":
            results = []
            for i, caption in enumerate(category):
                result, self.text_ids[i] = super(PersOwl, self).forward(obs, caption, text_ids=self.text_ids[i])
                results.append(result)
            results = torch.stack(results).sum(dim=0)
            return results
        else:
            raise ValueError("Invalid matching modality")
    
    def reset(self):
        self.captions = None
        self.text_ids = [None, None, None]
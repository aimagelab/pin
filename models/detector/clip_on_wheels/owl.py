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
        elif self.modality == "image-to-image":
            # similarities are computed on top of the pooled outputs of the encoders
            obs_inputs = self.processor(
                text=None, images=obs, return_tensors="pt", truncation=True
            )
            obs_outputs = self.model.base_model.vision_model(
                pixel_values=obs_inputs.pixel_values.to(self.model.device),
                return_dict=True,
            )
            obs_embeds = obs_outputs['pooler_output']
            obs_embeds = self.model.base_model.visual_projection(obs_embeds)
            obs_embeds = obs_embeds / torch.linalg.norm(obs_embeds, dim=-1, ord=2, keepdim=True)
            references = [to_pil_image(i[:,:,0:3]) for i in references]
            goal_inputs = self.processor(
                text=None, images=references, return_tensors="pt", truncation=True
            )
            goal_outputs = self.model.base_model.vision_model(
                pixel_values=goal_inputs.pixel_values.to(self.model.device),
                return_dict=True,
            )
            goal_embeds = goal_outputs['pooler_output']
            goal_embeds = self.model.base_model.visual_projection(goal_embeds)
            goal_embeds = goal_embeds / torch.linalg.norm(goal_embeds, dim=-1, ord=2, keepdim=True)
            
            logit_scale = self.model.base_model.logit_scale.exp()
            logits_per_ref = torch.matmul(goal_embeds, obs_embeds.t()) * logit_scale
            max_ref_index = logits_per_ref.argmax(dim=0)
            logits_per_image = logits_per_ref.t()
            output = OwlViTOutput(
                loss=None,
                logits_per_image=logits_per_image,
                logits_per_text=logits_per_ref,
                text_embeds=goal_embeds,
                image_embeds=obs_embeds,
                text_model_output=goal_outputs,
                vision_model_output=obs_outputs,
            )
            
            # the observation feature map is built on top of the last hidden state output
            
            last_hidden_state = output.vision_model_output['last_hidden_state']
            image_embeds = self.model.base_model.vision_model.post_layernorm(last_hidden_state)
            
            new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
            class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

            image_embeds = image_embeds[:, 1:, :] * class_token_out
            image_embeds = self.model.layer_norm(image_embeds)
            
            new_size = (
                image_embeds.shape[0],
                int(np.sqrt(image_embeds.shape[1])),
                int(np.sqrt(image_embeds.shape[1])),
                image_embeds.shape[-1],
            )
            image_embeds = image_embeds.reshape(new_size)
            query_embeds = output['text_embeds']
            
            feature_map = image_embeds
            text_outputs = output.text_model_output
            vision_outputs = output.vision_model_output
            
            batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
            image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))
                        
            query_embeds = query_embeds[max_ref_index, :]
            query_embeds = query_embeds.reshape(batch_size, 1, query_embeds.shape[-1])
                        
            query_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=query_embeds.device)

            (pred_logits, class_embeds) = self.model.class_predictor(image_feats, query_embeds, query_mask)
            
            pred_boxes = self.model.box_predictor(image_feats, feature_map)
            outputs = OwlViTObjectDetectionOutput(
                image_embeds=feature_map,
                text_embeds=query_embeds,
                pred_boxes=pred_boxes,
                logits=pred_logits,
                class_embeds=class_embeds,
                text_model_output=text_outputs,
                vision_model_output=vision_outputs,
            )
            
            results = post_process(
                outputs=outputs,
                target_sizes=torch.tensor(
                    [
                        [224.0, 224.0],
                    ]
                ),
            )
            boxes, scores = results[0]["boxes"], results[0]["scores"]
            image_relevance = torch.zeros((224, 224))
            
            for box, score in zip(boxes, scores):
                if score >= self.threshold:
                    box = [int(round(i, 2)) for i in box.tolist()]

                    if self.center_only:
                        u = int(round((box[1] + box[3]) / 2, 2))
                        v = int(round((box[0] + box[2]) / 2, 2))

                        image_relevance[u, v] = 1.0

                    else:
                        image_relevance[box[1] : box[3], box[0] : box[2]] = 1.0

            return image_relevance
        elif self.modality == "text-to-image":
            if self.captions is None:
                n, h, w, c = references.shape
                references = torch.from_numpy(references).to(self.device)
                masks = references[:, :, :, 3]
                references = references[:, :, :, :3]
                references, masks = squared_crop(references, masks)
                inputs = self.blip_model.processor(
                    images=references,
                    text=["a photo of"]*n,
                    return_tensors="pt"
                ).to(self.device)
                outputs = self.blip_model.model.generate(**inputs)
                self.captions = [self.blip_model.processor.decode(elem, skip_special_tokens=True) for elem in outputs]
            inputs = self.processor(
                text=self.captions, images=obs, return_tensors="pt", truncation=True
            )
            for k in inputs:
                inputs[k] = inputs[k].to(self.device)
            outputs = self.model(**inputs)
            results = post_process(
                outputs=outputs,
                target_sizes=torch.tensor(
                    [
                        [224.0, 224.0],
                    ]
                ),
            )
            boxes, scores = results[0]["boxes"], results[0]["scores"]
            image_relevance = torch.zeros((224, 224))
            for box, score in zip(boxes, scores):
                if score >= self.threshold:
                    box = [int(round(i, 2)) for i in box.tolist()]

                    if self.center_only:
                        u = int(round((box[1] + box[3]) / 2, 2))
                        v = int(round((box[0] + box[2]) / 2, 2))

                        image_relevance[u, v] = 1.0

                    else:
                        image_relevance[box[1] : box[3], box[0] : box[2]] = 1.0

            return image_relevance
        else:
            raise ValueError("Invalid matching modality")
    
    def reset(self):
        self.captions = None
        self.text_ids = [None, None, None]
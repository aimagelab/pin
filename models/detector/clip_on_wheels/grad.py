import torch
from typing import List
from torch import device
from torchvision.transforms.functional import pil_to_tensor

from utils.cow_utils.src.models.localization.clip_grad import ClipGrad
from utils.cow_utils.src.clip import clip
from models.detector.clip_on_wheels.utils import squared_crop

from utils.cow_utils.src.shared.utils import zeroshot_classifier

class PersGrad(ClipGrad):
    def __init__(self,
            clip_model_name: str,
            classes: List[str],
            classes_clip: List[str],
            templates: List[str],
            threshold: float,
            device: device,
            center_only: bool = False,
            modality: str = "category"):
        super(PersGrad, self).__init__(
            clip_model_name,
            classes,
            classes_clip,
            templates,
            threshold,
            device,
            center_only
        )
        self.modality = modality
        if self.modality == "text-to-image":
            self.blip_model = BlipModel(device=device)
        self.ref_embeds = [None, None, None]

    def forward(self, obs, references, category=None):
        if self.modality == "category":
            return super(PersGrad, self).forward(obs, category)
        elif self.modality == "captions":
            results = []
            if self.ref_embeds == [None, None, None]:
                self.ref_embeds = zeroshot_classifier(self.model, category, self.templates, self.device)
            for i, caption in enumerate(category):
                result = super(PersGrad, self).forward(pil_to_tensor(obs).unsqueeze(0), caption, ref_embed=self.ref_embeds[:, i].unsqueeze(-1))
                results.append(result)
            results = torch.stack(results).sum(dim=0)
            return results

        elif self.modality == "image-to-image":
            with torch.no_grad():
                references = torch.from_numpy(references).to(self.device)
                references = references[:, :, :, :3]
                ref_features = self.model.encode_image(references)
                ref_features = ref_features / ref_features.norm(dim=-1, keepdim=True)
                ref_features = ref_features.mean(dim=0)
                ref_features = ref_features.unsqueeze(-1).float()
                x = pil_to_tensor(obs).to(self.device).unsqueeze(0)
            obs_features = self.model.encode_image(x)
            obs_features = obs_features / obs_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100. * obs_features @ captions_embeds
            return self.interpret_vit(x, logits_per_image, self.model, self.device)
        elif self.modality == "text-to-image":
            n, h, w, c = references.shape
            references = torch.from_numpy(references).to(self.device)
            masks = references[:, :, :, 3]
            references = references[:, :, :, :3]
            references[~(masks.bool().unsqueeze(-1).repeat(1,1,1,3))] = 255
            references, masks = squared_crop(references, masks)
            inputs = self.blip_model.processor(
                images=references,
                text=["a photo of"]*n,
                return_tensors="pt"
            ).to(self.device)
            outputs = self.blip_model.model.generate(**inputs)
            captions = [self.blip_model.processor.decode(elem, skip_special_tokens=True) for elem in outputs]
            with torch.no_grad():
                tokenized_captions = clip.tokenize(captions).to(self.device)
                captions_embeds = self.model.encode_text(tokenized_captions)
                captions_embeds /= captions_embeds.norm(dim=-1, keepdim=True)
                captions_embeds = captions_embeds.mean(dim=0)
                captions_embeds /= captions_embeds.norm()
                captions_embeds = captions_embeds.unsqueeze(-1).float()
                x = pil_to_tensor(obs).to(self.device).unsqueeze(0)
            obs_features = self.model.encode_image(x)
            obs_features = obs_features / obs_features.norm(dim=-1, keepdim=True)
            logits_per_image = 100. * obs_features @ captions_embeds
            return self.interpret_vit(x, logits_per_image, self.model, self.device)
        
    def reset(self):
        self.ref_embeds = [None, None, None]
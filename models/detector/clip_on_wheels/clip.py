import torch
import numpy as np
import requests
import open_clip
from transformers import CLIPProcessor, CLIPModel

class ClipMatcher(torch.nn.Module):
    def __init__(self,
                model_name,
                model_weights,
                templates,
                device="cuda",
                resize_dim=224,
                match_threshold=0.5):
        super(ClipMatcher, self).__init__()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, model_weights)
        self.model.visual.output_tokens = True
        self.model.eval()
        self.device = device
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.resize_dim = resize_dim
        self.match_threshold = match_threshold
        self.templates = templates
        self.text_embeds = None
        
    @torch.no_grad()
    def reset(self):
        self.text_embeds = None
        
    @torch.no_grad()    
    def forward(self, obs, references, category=None):
        if self.text_embeds is None:
            input_captions = []
            for caption in category:
                current_captions = [template.format(caption) for template in self.templates]
                input_captions.extend(current_captions)
            inputs = self.tokenizer(input_captions).to(self.device)
            self.text_embeds = self.model.encode_text(inputs).reshape(len(category), len(self.templates), -1).mean(dim=1)
            self.text_embeds = self.text_embeds / self.text_embeds.norm(dim=-1, keepdim=True)
        image_embeds = self.preprocess(obs).unsqueeze(0).to(self.device)
        image_embeds = self.model.encode_image(image_embeds)[1] @ self.model.visual.proj
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        similarities = image_embeds @ self.text_embeds.T
        # print(similarities.max().item())
        matches = similarities > self.match_threshold
        
        return matches[0].cpu()
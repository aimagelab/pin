import torch
import clip
from PIL import Image

class ClipModel():
    def __init__(self, model_name):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def encode_image(self, img, normalize=False):
        image = self.preprocess(Image.open(img)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            if normalize:
                image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features
    
    def encode_text(self, text, normalize=False):
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        if normalize:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

if __name__ == "__main__":
    clip_enc = ClipModel("ViT-B/32") # RN50, RN101, RN50x4, RN50x16, RN50x64, ViT-B/32, ViT-B/16, ViT-L/14
    img_feat = clip_enc.encode_image("example_imgs/dog.png")
    text_feat = clip_enc.encode_text(["a dog"])
    print(img_feat)
    print(text_feat)
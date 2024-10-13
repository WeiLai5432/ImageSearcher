import torch
import clip
from PIL import Image
from utils import timer


class CLIPModel:
    @timer("init clip model")
    def __init__(self, model_name="ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print("clip model loaded")

    # @timer()
    def encode_image(self, image_path):
        if isinstance(image_path, str):
            image = Image.open(image_path)
        elif isinstance(image_path, Image.Image):
            image = image_path
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
        return image_features[0].tolist()

    # @timer()
    def encode_text(self, text):
        text = clip.tokenize(text).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text)
        return text_features[0].tolist()

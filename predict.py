from cog import BasePredictor, Input
from transformers import pipeline

class Predictor(BasePredictor):
    def setup(self):
        self.pipe = pipeline("text-to-image", model="CompVis/stable-diffusion-v1-4")

    def predict(self, prompt: str = Input(description="Enter a prompt")) -> str:
        image = self.pipe(prompt)[0]["image"]
        output_path = "/tmp/output.png"
        image.save(output_path)
        return output_path
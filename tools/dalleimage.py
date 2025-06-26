import time
import requests
from io import BytesIO
from PIL import Image
from pathlib import Path

from tools import BaseTool

class DalleEditTool(BaseTool):
    def __init__(self, config: dict):
        super().__init__(config)
        self.api_key = config.get('api_key') or self._get_api_key()
        self.size = config.get('size', '1024x1024')
        self.num_images = config.get('n', 1)

    def _get_api_key(self):
        """Get API key from environment variables"""
        import os
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise ValueError("API key not found in config or environment variables")
        return key

    def load_model(self):
        """No local model to load for API-based tool"""
        pass

    def process(self, image: Image, target_object: str) -> dict:
    
        image = self._ensure_rgba(image)
     
        img_bytes = self._pil_to_bytes(image)

        start_time = time.time()
        
        url = "https://api.openai.com/v1/images/edits"
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "image": ("input.png", img_bytes, "image/png")
        }
        data = {
            "prompt": target_object,
            "n": str(self.num_images),
            "size": self.size
        }

        response = requests.post(url, headers=headers, files=files, data=data, timeout=30)
        end_time = time.time()
        execution_time = end_time - start_time

        if response.status_code != 200:
            raise Exception(f"API Error {response.status_code}: {response.text}")
 
        resp_json = response.json()
        if "data" not in resp_json or not resp_json["data"]:
            raise Exception("API returned an empty response or no data.")

        image_url = resp_json["data"][0]["url"]
        edited_image_bytes = requests.get(image_url, timeout=30).content
        output_image = Image.open(BytesIO(edited_image_bytes))

        return {
            "image": output_image,
            "execution_time": execution_time
        }

    def _ensure_rgba(self, image: Image) -> Image:
        """Ensure the image is in RGBA mode."""
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        return image

    def _ensure_l_or_rgba(self, image: Image) -> Image:
        """
        Ensure the mask is in a valid format (L or RGBA).
        If it's RGB, convert it to L (grayscale).
        """
        if image.mode not in ["L", "LA", "RGBA"]:
            image = image.convert("L") 
        return image

    def _pil_to_bytes(self, image: Image) -> bytes:
        """Convert a PIL image to in-memory PNG bytes."""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

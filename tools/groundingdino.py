import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import groundingdino.datasets.transforms as T
from tools import BaseTool
import re 
import time

class GroundingDINOTool(BaseTool):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.config_path = config['config_path']
        self.checkpoint_path = config['checkpoint']
        self.box_threshold = config.get('box_threshold', 0.35)
        self.text_threshold = config.get('text_threshold', 0.25)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.load_model()

    def load_model(self):

        from groundingdino.util.slconfig import SLConfig
        from groundingdino.models import build_model
        from groundingdino.util.utils import clean_state_dict
        
        if not Path(self.checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint missing: {self.checkpoint_path}")
        if not Path(self.config_path).exists():
            raise FileNotFoundError(f"Config missing: {self.config_path}")
        
        args = SLConfig.fromfile(self.config_path)
        args.device = self.device
        
        self.model = build_model(args)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.model.eval().to(self.device)

    def process(self, image: Image.Image, subtask_name: str, **kwargs) -> dict:

        if image.mode != "RGB":
            image = image.convert("RGB")

        text_prompt = kwargs.get("from_object", "object.")
        text_prompt = text_prompt.strip() + "."

        image_tensor, _ = self.transform(image, None)
        image_tensor = image_tensor.to(self.device)

        boxes, phrases, execution_time = self._predict_boxes(image_tensor, text_prompt)

        annotated_image, bounding_boxes = self._annotate_and_collect(
            image_pil=image.copy(),
            boxes=boxes,
            phrases=phrases,
            from_object=kwargs.get("from_object", None)
        )

        # print("GDINO bboxes: ", bounding_boxes)

        if subtask_name.lower() == "object detection":
            return {"image": annotated_image, "bounding_boxes": bounding_boxes, "execution_time": execution_time}
        else:
            return {"image": image, "bounding_boxes": bounding_boxes, "instance_count": len(bounding_boxes), "execution_time": execution_time}


    def _predict_boxes(self, image_tensor, text_prompt):
        from groundingdino.util.utils import get_phrases_from_posmap
        from groundingdino.util.vl_utils import create_positive_map_from_span

        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(image_tensor.unsqueeze(0), captions=[text_prompt.lower()])
            end_time = time.time()
        
        logits = outputs["pred_logits"].sigmoid()[0]  
        boxes = outputs["pred_boxes"][0]  
        execution_time = end_time - start_time       

        max_prob = logits.max(dim=1)[0]
        keep = max_prob > self.box_threshold
        boxes = boxes[keep].cpu()
        logits = logits[keep].cpu()

        tokenized = self.model.tokenizer(text_prompt)
        phrases = []
        for logit in logits:
            on_tokens = logit > self.text_threshold
            phrase = get_phrases_from_posmap(on_tokens, tokenized, self.model.tokenizer)
            phrases.append(phrase)

        return boxes, phrases, execution_time

    def _annotate_and_collect(self, image_pil, boxes, phrases, from_object=None):
        draw = ImageDraw.Draw(image_pil)
        W, H = image_pil.size
        
        bounding_boxes = []
        if from_object is not None:
            from_object_regex = re.compile(re.escape(from_object), re.IGNORECASE)
        else:
            from_object_regex = None

        for box, phrase in zip(boxes, phrases):

            cx, cy, w, h = box
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            x2 = (cx + w/2) * W
            y2 = (cy + h/2) * H
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            if from_object_regex is not None:
                if not from_object_regex.search(phrase):
                    continue

            color = tuple(np.random.randint(0, 255, size=3).tolist())

            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except IOError:
                font = ImageFont.load_default()
            bbox = draw.textbbox((x1, y1), phrase, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]         
            draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
            draw.text((x1, y1 - text_h), phrase, fill="white", font=font)

            bounding_boxes.append([x1, y1, x2, y2])

        return image_pil, bounding_boxes

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor
from tools import BaseTool
import os
import time

# A dictionary of CSS3 color names and their corresponding RGB values.
CSS3_COLORS = {
    "aliceblue": (240, 248, 255), "antiquewhite": (250, 235, 215), "aqua": (0, 255, 255),
    "aquamarine": (127, 255, 212), "azure": (240, 255, 255), "beige": (245, 245, 220),
    "bisque": (255, 228, 196), "black": (0, 0, 0), "blanchedalmond": (255, 235, 205),
    "blue": (0, 0, 255), "blueviolet": (138, 43, 226), "brown": (165, 42, 42),
    "burlywood": (222, 184, 135), "cadetblue": (95, 158, 160), "chartreuse": (127, 255, 0),
    "chocolate": (210, 105, 30), "coral": (255, 127, 80), "cornflowerblue": (100, 149, 237),
    "cornsilk": (255, 248, 220), "crimson": (220, 20, 60), "cyan": (0, 255, 255),
    "darkblue": (0, 0, 139), "darkcyan": (0, 139, 139), "darkgoldenrod": (184, 134, 11),
    "darkgray": (169, 169, 169), "darkgreen": (0, 100, 0), "darkgrey": (169, 169, 169),
    "darkkhaki": (189, 183, 107), "darkmagenta": (139, 0, 139), "darkolivegreen": (85, 107, 47),
    "darkorange": (255, 140, 0), "darkorchid": (153, 50, 204), "darkred": (139, 0, 0),
    "darksalmon": (233, 150, 122), "darkseagreen": (143, 188, 143), "darkslateblue": (72, 61, 139),
    "darkslategray": (47, 79, 79), "darkslategrey": (47, 79, 79), "darkturquoise": (0, 206, 209),
    "darkviolet": (148, 0, 211), "deeppink": (255, 20, 147), "deepskyblue": (0, 191, 255),
    "dimgray": (105, 105, 105), "dimgrey": (105, 105, 105), "dodgerblue": (30, 144, 255),
    "firebrick": (178, 34, 34), "floralwhite": (255, 250, 240), "forestgreen": (34, 139, 34),
    "fuchsia": (255, 0, 255), "gainsboro": (220, 220, 220), "ghostwhite": (248, 248, 255),
    "gold": (255, 215, 0), "goldenrod": (218, 165, 32), "gray": (128, 128, 128),
    "green": (0, 128, 0), "greenyellow": (173, 255, 47), "grey": (128, 128, 128),
    "honeydew": (240, 255, 240), "hotpink": (255, 105, 180), "indianred": (205, 92, 92),
    "indigo": (75, 0, 130), "ivory": (255, 255, 240), "khaki": (240, 230, 140),
    "lavender": (230, 230, 250), "lavenderblush": (255, 240, 245), "lawngreen": (124, 252, 0),
    "lemonchiffon": (255, 250, 205), "lightblue": (173, 216, 230), "lightcoral": (240, 128, 128),
    "lightcyan": (224, 255, 255), "lightgoldenrodyellow": (250, 250, 210), "lightgray": (211, 211, 211),
    "lightgreen": (144, 238, 144), "lightgrey": (211, 211, 211), "lightpink": (255, 182, 193),
    "lightsalmon": (255, 160, 122), "lightseagreen": (32, 178, 170), "lightskyblue": (135, 206, 250),
    "lightslategray": (119, 136, 153), "lightslategrey": (119, 136, 153), "lightsteelblue": (176, 196, 222),
    "lightyellow": (255, 255, 224), "lime": (0, 255, 0), "limegreen": (50, 205, 50),
    "linen": (250, 240, 230), "magenta": (255, 0, 255), "maroon": (128, 0, 0),
    "mediumaquamarine": (102, 205, 170), "mediumblue": (0, 0, 205), "mediumorchid": (186, 85, 211),
    "mediumpurple": (147, 112, 219), "mediumseagreen": (60, 179, 113), "mediumslateblue": (123, 104, 238),
    "mediumspringgreen": (0, 250, 154), "mediumturquoise": (72, 209, 204), "mediumvioletred": (199, 21, 133),
    "midnightblue": (25, 25, 112), "mintcream": (245, 255, 250), "mistyrose": (255, 228, 225),
    "moccasin": (255, 228, 181), "navajowhite": (255, 222, 173), "navy": (0, 0, 128),
    "oldlace": (253, 245, 230), "olive": (128, 128, 0), "olivedrab": (107, 142, 35),
    "orange": (255, 165, 0), "orangered": (255, 69, 0), "orchid": (218, 112, 214),
    "palegoldenrod": (238, 232, 170), "palegreen": (152, 251, 152), "paleturquoise": (175, 238, 238),
    "palevioletred": (219, 112, 147), "papayawhip": (255, 239, 213), "peachpuff": (255, 218, 185),
    "peru": (205, 133, 63), "pink": (255, 192, 203), "plum": (221, 160, 221),
    "powderblue": (176, 224, 230), "purple": (128, 0, 128), "red": (255, 0, 0),
    "rosybrown": (188, 143, 143), "royalblue": (65, 105, 225), "saddlebrown": (139, 69, 19),
    "salmon": (250, 128, 114), "sandybrown": (244, 164, 96), "seagreen": (46, 139, 87),
    "seashell": (255, 245, 238), "sienna": (160, 82, 45), "silver": (192, 192, 192),
    "skyblue": (135, 206, 235), "slateblue": (106, 90, 205), "slategray": (112, 128, 144),
    "slategrey": (112, 128, 144), "snow": (255, 250, 250), "springgreen": (0, 255, 127),
    "steelblue": (70, 130, 180), "tan": (210, 180, 140), "teal": (0, 128, 128),
    "thistle": (216, 191, 216), "tomato": (255, 99, 71), "turquoise": (64, 224, 208),
    "violet": (238, 130, 238), "wheat": (245, 222, 179), "white": (255, 255, 255),
    "whitesmoke": (245, 245, 245), "yellow": (255, 255, 0), "yellowgreen": (154, 205, 50)
}

def get_closest_color_name_from_custom_map(rgb_tuple: tuple) -> str:
    """
    Finds the closest CSS3 color name for a given RGB tuple from a custom map.
    """
    min_distance = float('inf')
    closest_color_name = 'unknown'
    
    for name, color_rgb in CSS3_COLORS.items():
        distance = np.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(rgb_tuple, color_rgb)]))
        
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
            
    return closest_color_name

class SAMTool(BaseTool):
    def __init__(self, config: dict):
        super().__init__(config)
        self.model = None
        self.predictor = None
        self.model_type = config.get('model_type', 'vit_h')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.multimask_output = config.get('multimask_output', True)
        self.checkpoint = config.get('checkpoint', None)

    def load_model(self, checkpoint_path: str):
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found at {checkpoint_path}")
        self.model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model.to(device=self.device)
        self.predictor = SamPredictor(self.model)

    def process(
        self,
        image: Image.Image,
        subtask_name: str,
        bounding_boxes: list,
        random_color: bool = True,
        alpha: float = 0.6
    ) -> dict:

        if not bounding_boxes:
            return {
                "image": None, "cutout_images": [], "execution_time": 0,
                "mask_size": [], "relative_mask_area": [], "rgb_color": [],
                "text_color": []
            }

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.predictor is None:
            if self.checkpoint is None:
                raise RuntimeError("No checkpoint path provided. Please specify one in config.")
            self.load_model(self.checkpoint)

        image_np = np.array(image)
        self.predictor.set_image(image_np)

        h, w = image_np.shape[:2]
        image_area = h * w

        boxes_torch = torch.tensor(bounding_boxes, dtype=torch.float, device=self.device)
        if boxes_torch.ndim != 2 or boxes_torch.shape[1] != 4:
            raise ValueError("bounding_boxes must be a list of [x1, y1, x2, y2] entries.")

        boxes_torch = self.predictor.transform.apply_boxes_torch(boxes_torch, image_np.shape[:2])

        start_time = time.time()
        masks, scores, logits = self.predictor.predict_torch(
            point_coords=None, point_labels=None, boxes=boxes_torch,
            multimask_output=self.multimask_output
        )
        end_time = time.time()

        best_masks = masks[:, 0, :, :] if self.multimask_output else masks.squeeze(1)

        mask_sizes, relative_mask_areas, rgb_colors, text_colors = [], [], [], []

        for mask_tensor in best_masks:
            mask_size_pixels = torch.sum(mask_tensor).item()
            mask_sizes.append(int(mask_size_pixels))

            relative_area = (mask_size_pixels / image_area) if image_area > 0 else 0
            relative_mask_areas.append(relative_area)

            binary_mask_np = mask_tensor.cpu().numpy()
            item_pixels = image_np[binary_mask_np]

            if item_pixels.size > 0:
                avg_color = np.mean(item_pixels, axis=0)
                avg_color_int_tuple = tuple(int(c) for c in avg_color)
                avg_color_int_list = list(avg_color_int_tuple)
            else:
                avg_color_int_tuple = (0, 0, 0)
                avg_color_int_list = [0, 0, 0]

            rgb_colors.append(avg_color_int_list)
            
            color_name = get_closest_color_name_from_custom_map(avg_color_int_tuple)
            text_colors.append(color_name)

        final_image, cutout_images = overlay_and_generate_cutouts(
            image, best_masks.cpu().numpy(), subtask_name, random_color, alpha
        )
    
        execution_time = end_time - start_time

        return {
            "image": final_image, "cutout_images": cutout_images,
            "execution_time": execution_time, "mask_size": mask_sizes,
            "relative_mask_area": relative_mask_areas, "rgb_color": rgb_colors,
            "text_color": text_colors
        }

def overlay_and_generate_cutouts(
    original_image: Image.Image,
    mask_tensor: np.ndarray,
    subtask_name: str,
    random_color: bool = False,
    alpha: float = 0.4
) -> tuple:

    image_rgba = original_image.convert("RGBA")
    image_np = np.array(image_rgba)

    B, H, W = mask_tensor.shape
    if image_np.shape[0] != H or image_np.shape[1] != W:
        raise ValueError("Mask dimension does not match image dimension.")

    modified_image = image_np.copy()
    cutout_images = []
    for b_idx in range(B):
        single_mask = mask_tensor[b_idx, :, :]

        if single_mask.dtype == np.bool_:
            single_mask = single_mask.astype(np.uint8) * 255

        mask_bool = single_mask > 0
        cutout_np = np.zeros_like(image_np, dtype=np.uint8)
        cutout_np[mask_bool] = image_np[mask_bool]
        cutout_np[~mask_bool, 3] = 0

        cutout_image = Image.fromarray(cutout_np)
        cutout_images.append(cutout_image)

        if subtask_name.lower() == "object segmentation":
            if random_color:
                color = np.random.randint(0, 256, size=3).tolist()
            else:
                color = [30, 144, 255]

            overlay_color = np.array(color + [int(alpha * 255)])

            ys, xs = np.where(mask_bool)
            modified_image[ys, xs, :3] = (
                (1 - alpha) * modified_image[ys, xs, :3].astype(np.float32)
                + alpha * overlay_color[:3]
            ).astype(np.uint8)
            modified_image[ys, xs, 3] = 255

        else:
            modified_image[mask_bool, :3] = [255, 255, 255]
            modified_image[mask_bool, 3] = 0

    return Image.fromarray(modified_image), cutout_images
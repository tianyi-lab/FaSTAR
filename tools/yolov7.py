import sys
sys.path.append("/cmlscratch/vnkiran/COSTA-Code/test/tools/yolov7")
import torch
import cv2
import numpy as np
import time
import random
import math
import os
from PIL import Image, ImageDraw, ImageFont
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from importlib import import_module
from tools import BaseTool  

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2 
    y[:, 1] = x[:, 1] - x[:, 3] / 2 
    y[:, 2] = x[:, 0] + x[:, 2] / 2  
    y[:, 3] = x[:, 1] + x[:, 3] / 2  
    return y

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None

def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(w, map_location=map_location)
        mdl = ckpt['ema' if ckpt.get('ema') else 'model'].float()
        mdl = mdl.eval()
        model.append(mdl)

    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = True
        elif isinstance(m, nn.Upsample):
            m.recompute_scale_factor = None
        elif isinstance(m, Conv):
            m._non_persistent_buffers_set = set()
    
    return model[-1] if len(model) == 1 else model

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    new_size = make_divisible(img_size, int(s))
    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False):
    nc = prediction.shape[2] - 5
    xc = prediction[..., 4] > conf_thres

    max_det = 300
    max_nms = 30000
    time_limit = 10.0
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    
    t = time.time()
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        if nc == 1:
            x[:, 5:] = x[:, 4:5]
        else:
            x[:, 5:] *= x[:, 4:5]
        
        box = xywh2xyxy(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
        
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        
        boxes, scores = x[:, :4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break
    
    return output

def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    return coords

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    tl = max(1, line_thickness)
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tl - 1)[0]
        c2_label = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2_label, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255],
                    thickness=tl - 1, lineType=cv2.LINE_AA)

def select_device(device=''):
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:
        os.environ['CUDA_VISIBLE_DEVICES'] = device
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'
    cuda = not cpu and torch.cuda.is_available()
    return torch.device('cuda:0' if cuda else 'cpu')

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = im.shape[:2] 
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


class YOLOTool(BaseTool):
   
    def __init__(self, config: dict):
        super().__init__(config)
        self.checkpoint = config.get('checkpoint', None)
        if self.checkpoint is None:
            raise ValueError("YOLOTool requires a 'checkpoint' parameter.")
        self.conf_thres = config.get('conf_thres', 0.25)
        self.iou_thres = config.get('iou_thres', 0.45)
        self.img_size = config.get('img_size', 640)
        self.classes = config.get('classes', None)
        self.agnostic_nms = config.get('agnostic_nms', False)
        self.model = None
        self.device = None
        self.half = False
        self.stride = 32
        self.names = None
        self.colors = None
        self.load_model(self.checkpoint)

    def load_model(self, checkpoint_path: str):
        self.device = select_device('')
        self.half = self.device.type != 'cpu'
        self.model = attempt_load(checkpoint_path, map_location=self.device)
        self.names  = self.model.names
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.img_size, s=self.stride)
        if self.half:
            self.model.half()
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.device.type != 'cpu':
            dummy_input = torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(self.model.parameters()))
            self.model(dummy_input)

    def preprocess(self, img):
        img, _, _ = letterbox(img, new_shape=self.img_size, stride=self.stride)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def process(self, image: Image, subtask_name: str, **kwargs):

        if image.mode != "RGB":
            image = image.convert("RGB")
  
        target_class = kwargs.get("from_object", None)
        if target_class is not None:
            target_class = target_class.lower()
        # print("YOLOv7 Target Class: ", target_class)

        im0 = np.array(image)
        im0 = im0[:, :, ::-1].copy()
        img = self.preprocess(im0.copy())

        with torch.no_grad():
            start_time = time.time()
            pred = self.model(img, augment=False)[0]
            end_time = time.time() 

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                classes=self.classes, agnostic=self.agnostic_nms)

        filtered_boxes = []
        det = pred[0]  
        
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_name = self.names[int(cls)]
                if target_class is not None and class_name.lower() != target_class:
                    continue
                box_coords = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                filtered_boxes.append(box_coords)
                if subtask_name.lower() == "object detection":
                    plot_one_box(xyxy, im0, label=class_name, color=self.colors[int(cls)], line_thickness=2)

        if subtask_name.lower() == "object detection":
            result_image = Image.fromarray(cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)) 
        else:
            result_image = image 
        
        execution_time = end_time - start_time

        # print("YOLO bboxes: ", filtered_boxes)
        # print("YOLO Classes: ", self.names)

        return {"image": result_image, "bounding_boxes": filtered_boxes, "instance_count": len(filtered_boxes), "execution_time": execution_time}

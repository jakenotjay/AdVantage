from re import M
from ..pipeline import PipelineHandler
from ..sendables import VideoProcessingFrame
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (non_max_suppression, scale_coords)
from yolov5.utils.augmentations import letterbox   
from ..lib import Prediction   


class YoloProcessor(PipelineHandler):
    model = None
    device = None
    half = None
    imgz = None
    stride = 0
    conf_thres=0.5  # confidence threshold
    iou_thres=0.45  # NMS IOU threshold
    max_det=1000  # maximum detections per image
    classes=None  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False  # class-agnostic NMS
    skip_frames = 0
    frame_count = 0
    def __init__(
        self, 
        weights,
        imgz = None,
        stride = 32, 
        device='',
        conf_thres=0.6,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        half=False,  # use FP16 half-precision inference
        skip_frames = 0,
        clean_predictions_after_frame = True,
        first_frame_only= False
    ) -> None:
        super().__init__()
        self.device = device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.imgz = imgz
        self.stride = stride
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms 
        self.half = half
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.clean_predictions_after_frame = clean_predictions_after_frame
        self.first_frame_only = first_frame_only
            
    def handle(self, task: VideoProcessingFrame, next):
        if task.frame_id > 0 and self.first_frame_only:
            return next(task)

        if self.skip_frames > 0 and task.frame_id > 0: 
            if self.frame_count < self.skip_frames:
                self.frame_count += 1
                return next(task)
            else:
                self.frame_count = 0 
        else:
            self.frame_count += 1            

        if self.imgz == None:
            self.imgz = (task.frame_width, task.frame_height)
        imgz = self.imgz
        model = self.model
        if task.frame_id == 0:
            # Half
            self.half &= (self.model.pt or self.model.jit or self.model.onnx or self.model.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
            if self.model.pt or self.model.jit:
                self.model.model.half() if self.half else self.model.model.float()

            self.model.warmup(imgsz=(1 if self.model.pt else 1, 3, *imgz))  # warmup
       
        if imgz[0] != task.frame_width and imgz[1] != task.frame_height:
            img = letterbox(task.frame, imgz, stride=self.stride, auto=True)[0]
        else:
            img = task.frame.copy()    

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img[None]

        pred = self.model(img)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        predictions = []
        # Process predictions
        for i, det in enumerate(pred):  # per image
            #gn = torch.tensor(task.frame.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], task.frame.shape).round()
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = model.names[c]
                x1 = xyxy[0].item()
                y1 = xyxy[1].item()
                x2 = xyxy[2].item()
                y2 = xyxy[3].item()
                prediction = Prediction(label, conf.item(),x1,y1,x2,y2)
                predictions.append(prediction)

        task.put('predictions', predictions)       
        result = next(task)  
        if self.clean_predictions_after_frame:
            task.put('predictions', None)
        return result    
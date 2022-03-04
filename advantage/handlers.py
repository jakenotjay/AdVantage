import cv2
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from vantage_api.geometry import VantageGeometry
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (non_max_suppression, scale_coords)
from .lib import Prediction

from yolov5.utils.augmentations import letterbox                           

class Verbose(PipelineHandler):
    def handle(self, task: VideoProcessingFrame, next):
        print('Processing Frame: '+str(task.frame_id))

        handledTask = next(task)

        if task.has('predictions'):
            for prediction in task.get('predictions'):
                print('\t Label: '+prediction.getLabel()+ ' Score: '+str(prediction.getScore())+' Box: '+str(prediction.getBox()))

        return handledTask

class FrameBuffer(PipelineHandler):
    buffer = []
    buffer_size = 0
    def __init__(self, buffer_size = 3) -> None:
        super().__init__()
        self.buffer_size = buffer_size

    def handle(self, task: VideoProcessingFrame, next):
        self.buffer.append(task)
        if len(self.buffer) > self.buffer_size: 
            self.buffer = self.buffer[-(self.buffer_size):]
        task.put('frame_buffer', self.buffer)
        return next(task)        

class VideoAttachGeoData(PipelineHandler):
    geo = None
    def __init__(self, geoFile) -> None:
        super().__init__()
        self.geo = VantageGeometry(geoFile)

    def handle(self, task: VideoProcessingFrame, next):
        task.put('geo', self.geo.getFrame(task.frame_id))
        return next(task)

class VideoWriter(PipelineHandler):
    video = None
    outputFile = None
    def __init__(self, outputFile) -> None:
        super().__init__()
        self.outputFile = outputFile

    def handle(self, task: VideoProcessingFrame, next):
        if self.video == None:
            self.video = cv2.VideoWriter(self.outputFile, cv2.VideoWriter_fourcc(*'DIVX'), task.fps, (task.frame_width, task.frame_height))
        task.put('output_frame', task.frame.copy())
        result = next(task)
        self.video.write(result.get('output_frame'))
        return result

    def release(self):
        if self.video != None:
            self.video.release()
        return self    

class VideoPredictionVisulisation(PipelineHandler):
    colour = None
    size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale = 1,
    fontColour = (0, 0, 255),
    fontThickness = 2

    def __init__(
        self,
        colour=(255, 0, 0),
        size= 2,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        fontColour = (0, 0, 255),
        fontThickness = 2
    ) -> None:
        super().__init__()
        self.colour = colour
        self.size = size
        self.font = font
        self.fontScale = fontScale
        self.fontColour = fontColour
        self.fontThickness = fontThickness

    def handle(self, task: VideoProcessingFrame, next):
       if task.has('output_frame') and task.has('predictions'):
            output_frame = task.get('output_frame')
            for prediction in task.get('predictions'):
               box = prediction.getBox()
               cv2.rectangle(output_frame, (box[0],box[1]),(box[2],box[3]), self.colour, self.size)
               self.printText(output_frame, prediction.getLabel() , (box[2] - 50,box[3] + 50))
               self.printText(output_frame, str(prediction.getScore()) , (box[2] - 100,box[3] + 100))
            task.put('output_frame', output_frame)
       return next(task)

    def printText(self,frame, text, position):
        cv2.putText(
            frame, 
            text, 
            position, 
            self.font, 
            self.fontScale, 
            self.fontColour, 
            self.fontThickness, 
            cv2.LINE_AA
        ) 


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
            
    def handle(self, task: VideoProcessingFrame, next):
        if self.imgz == None:
            self.imgz = (task.frame_width, task.frame_height)
        imgz = self.imgz
        model = self.model
        if task.frame_id == 0:
            # Half
            self.half &= (self.model.pt or self.model.jit or self.model.onnx or self.model.engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
            if self.model.pt or self.model.jit:
                self.model.model.half() if self.half else self.model.model.float()

            self.model.warmup(imgsz=(1 if self.model.pt else 1, 3, *imgz), half=self.half)  # warmup
       
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
                prediction = Prediction(label, conf.item(), xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item())
                predictions.append(prediction)

        task.put('predictions', predictions)       
        return next(task)  
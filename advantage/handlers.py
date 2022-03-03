import cv2
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from vantage_api.geometry import VantageGeometry
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from yolov5.utils.augmentations import letterbox                           

class Verbose(PipelineHandler):
    def handle(self, task: VideoProcessingFrame, next):
        print('Processing Frame: '+str(task.frame_id))
        handledTask = next(task)
        #print('this happens after all tasks')
        return handledTask

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
    def __init__(self, outputFile, width, height, fps) -> None:
        super().__init__()
        self.video = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))

    def handle(self, task: VideoProcessingFrame, next):
        result = next(task)
        self.video.write(result.frame)
        print(self.video)
        return result

    def release(self):
        self.video.release()
        return self    


class YoloProcessor(PipelineHandler):
    model = None
    device = None
    half = None
    def __init__(self, weights, device='', imgsz=(3072, 3072)) -> None:
        super().__init__()
        self.device = device = select_device(device)
        self.model = model = DetectMultiBackend(weights, device=self.device, dnn=False)
            
    def handle(self, task: VideoProcessingFrame, next):
        img = letterbox(task.frame, 640, stride=32, auto=True)[0]

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img[None]

        pred = self.model(img)
        print(pred)
        return next(task)  
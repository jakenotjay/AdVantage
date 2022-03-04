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


class YoloProcessor(PipelineHandler):
    model = None
    device = None
    half = None
    def __init__(self, weights, device='') -> None:
        super().__init__()
        self.device = device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
            
    def handle(self, task: VideoProcessingFrame, next):
        img = letterbox(task.frame, task.frame_width, stride=32, auto=True)[0]

        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        img = img[None]

        pred = self.model(img)
        print(pred)
        return next(task)  
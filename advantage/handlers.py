import cv2
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from vantage_api.geometry import VantageGeometry
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, time_sync

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
    def __init__(self, weights, device='') -> None:
        super().__init__()
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)

    def handle(self, task: VideoProcessingFrame, next):
        im = torch.from_numpy(task.frame).to(self.device)
        #im = im.float()
        #im /= 255  # 0 - 255 to 0.0 - 1.0
        #if len(im.shape) == 3:
        #    im = im[None]  # expand for batch dim
        results = self.model(im)
        print(results)
        #task.put('yolo', 'test') 
        #print(task.get('yolo').results())
        return next(task)  
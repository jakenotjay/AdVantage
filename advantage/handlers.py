import cv2
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from vantage_api.geometry import VantageGeometry
import torch
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

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
        stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine

        imgsz = check_img_size(imgsz, s=stride)  # check image size
        # Half
        self.half = half = (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            model.model.half() if half else model.model.float()

        bs=1
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz), half=half)  # warmup
            
    def handle(self, task: VideoProcessingFrame, next):
        im = torch.from_numpy(task.frame).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
    
        pred = self.model(im)
        print(pred)
        #task.put('yolo', 'test') 
        #print(task.get('yolo').results())
        return next(task)  
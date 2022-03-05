from asyncio import run
import cv2
from cv2 import line
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from vantage_api.geometry import VantageGeometry
import torch
import numpy as np
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.general import (non_max_suppression, scale_coords)
from .lib import Prediction, ProcessedVideo
import os
import math
from yolov5.utils.augmentations import letterbox      


class ObjectTracker(PipelineHandler):
    #Data persists between frames
    trackedObject = []
    def __init__(self) -> None:
        super().__init__()
        #any setup options here

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        #processing of frame here (task.frame). must always return next(task)
        result = next(task)  
        #any processing after the pipeline can be done here
        return result

class CloudDetector(PipelineHandler):
    #Data persists between frames
    clouds = []
    def __init__(self) -> None:
        super().__init__()
        #any setup options here

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        
        white = np.array([255, 255, 255])
        lowerBound = np.array([30,30,30])

        mask = cv2.inRange(task.frame, lowerBound, white)
        res = cv2.bitwise_and(task.frame, task.frame, mask=mask)

        cv2.imwrite('output/frame_'+str(task.frame_id)+'_cloud.jpg', res)
        
        result = next(task)  
        #any processing after the pipeline can be done here
        return result

    def after(self, processed: ProcessedVideo):
        print('processing cloud')    

class RunwayDetector(PipelineHandler):
    image_width = None
    output_test_images = False
    def __init__(self, image_width=None, output_test_images = False) -> None:
        super().__init__()
        self.lines = []
        self.image_width = image_width
        self.output_test_images = output_test_images

    def handle(self, task: VideoProcessingFrame, next):

        resize_img_width = self.image_width if self.image_width != None else task.frame_width
        img = task.frame.copy()
        scale = resize_img_width / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
        img_copy = img.copy()
        bin_img = self.convertImageToStandardisedBinary(img)
        lines = self.findLinesInBinaryImage(bin_img, width)
        img_copy = self.saveLinesToTask(task,img_copy, lines, width, height)
        img_copy, bin_img = self.saveTestImages(task, img_copy, bin_img)

        return next(task)   

    def saveTestImages(self, task, img, bin_img):
        if self.output_test_images:
            cv2.imwrite('output/frame_'+str(task.frame_id)+'_thresh.jpg', bin_img)
            cv2.imwrite('output/frame_'+str(task.frame_id)+'.jpg', img)   
        return img, bin_img    

    def saveLinesToTask(self,task,img, lines, width, height):
        if lines is not None:
            outputLines = []
            lines = sorted(lines, key=self.getLineLength)
            for i in range(0, len(lines)):
                l = lines[i][0]
                x1p = l[0] / width 
                x2p = l[2] / width
                y1p = l[1] / height
                y2p = l[3] / height
                outputLines.append([
                    int(task.frame_width * x1p),
                    int(task.frame_height * y1p),
                    int(task.frame_width * x2p),
                    int(task.frame_height * y2p),
                ])
                if self.output_test_images:
                    cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)  

            task.put('runways', outputLines)
        return img    

    def findLinesInBinaryImage(self, img, width):
        runway_length_drop_percentage = 10
        currentRunwayDropPercentage = runway_length_drop_percentage
        currentRunwayMin = width - (width * (currentRunwayDropPercentage / 100))
        lines = None

        while currentRunwayDropPercentage < 100:
            lines = cv2.HoughLinesP(img, 1, np.pi / 180, 150, None, currentRunwayMin, 5)
            if lines is not None:
                break  
            currentRunwayDropPercentage += runway_length_drop_percentage
            currentRunwayMin = width - (width * (currentRunwayDropPercentage / 100))

        return lines    

    def convertImageToStandardisedBinary(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

        white_count = np.sum(thresh == 255)
        black_count = np.sum(thresh == 0)
        if white_count > black_count:
            ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY_INV)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(thresh, rect_kernel, iterations = 1)

    def getLineLength(self, line):
        l = line[0]
        x = l[2] - l[0]
        y = l[3] - l[1]
        return math.sqrt(math.pow(x,2) * math.pow(y, 2))    

class Verbose(PipelineHandler):
    def handle(self, task: VideoProcessingFrame, next):
        print('Processing Frame: '+str(task.frame_id))

        handledTask = next(task)

        if task.has('predictions'):
            for prediction in task.get('predictions'):
                print('\t Label: '+prediction.getLabel()+ ' Score: '+str(prediction.getScore())+' Box: '+str(prediction.getBox()))

        return handledTask

class PipelineKiller(PipelineHandler):
    frames_to_process = 0
    def __init__(self, frames_to_process = 1) -> None:
        super().__init__()
        self.frames_to_process = frames_to_process

    def handle(self, task: VideoProcessingFrame, next):
        if self.frames_to_process > 0 and task.frame_id >= self.frames_to_process:
            task.continue_frames = False
            return task
        else:
            return next(task)        

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
    image_frame_output_dir = None
    file_prefix = ''
    output_video = True
    def __init__(self, output_file_name, output_video = True, image_frame_output_dir = None) -> None:
        super().__init__()
        self.outputFile = output_file_name
        self.image_frame_output_dir = image_frame_output_dir
        self.file_prefix = ''
        self.output_video = output_video
        if image_frame_output_dir != None:
            self.file_prefix = os.path.splitext(os.path.basename(output_file_name))[0]


    def handle(self, task: VideoProcessingFrame, next):
        if self.video == None and self.output_video:
            self.video = cv2.VideoWriter(self.outputFile, cv2.VideoWriter_fourcc(*'DIVX'), task.fps, (task.frame_width, task.frame_height)) 
        task.put('output_frame', task.frame.copy())
        result = next(task)
        if self.output_video:
            self.video.write(result.get('output_frame'))
        self.imageWriter(result)
        return result

    def imageWriter(self, task:VideoProcessingFrame):
        if self.image_frame_output_dir != None:
            imgPath = os.path.join(self.image_frame_output_dir, str(task.frame_id)+'_'+self.file_prefix+'.jpg')
            cv2.imwrite(imgPath, task.get('output_frame'))


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
    include = []

    def __init__(
        self,
        colour=(255, 0, 0),
        size= 2,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        fontColour = (0, 0, 255),
        fontThickness = 2,
        include = []
    ) -> None:
        super().__init__()
        self.colour = colour
        self.size = size
        self.font = font
        self.fontScale = fontScale
        self.fontColour = fontColour
        self.fontThickness = fontThickness
        self.include = include

    def handle(self, task: VideoProcessingFrame, next):
        if task.has('output_frame'):
            output_frame = task.get('output_frame')
            if self.processParam(task, 'predictions'):
                for prediction in task.get('predictions'):
                    box = prediction.getBox()
                    cv2.rectangle(output_frame, (box[0],box[1]),(box[2],box[3]), self.colour, self.size)
                    self.printText(output_frame, prediction.getLabel() , (box[2] - 50,box[3] + 50))
                    self.printText(output_frame, str(prediction.getScore()) , (box[2] - 100,box[3] + 100))
            if self.processParam(task, 'runways'):
                lines = task.get('runways')
                if lines is not None:
                    for i in range(0, len(lines)):
                        l = lines[i]
                        cv2.line(output_frame, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)   
            task.put('output_frame', output_frame)
        return next(task)

    def processParam(self,task: VideoProcessingFrame, param):
        return (len(self.include) == 0 or param in self.include) and task.has(param)

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
        skip_frames = 0
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
            
    def handle(self, task: VideoProcessingFrame, next):

        if self.skip_frames > 0 and self.frame_count > 0: 
            if self.frame_count > 0 and self.frame_count <= self.skip_frames:
                if self.frame_count >= self.skip_frames:
                    self.frame_count = 0
                else:
                   self.frame_count += 1 
                return next(task)
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
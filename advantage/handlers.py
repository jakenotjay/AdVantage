from re import M
import cv2
from .pipeline import PipelineHandler
from .sendables import VideoProcessingFrame
from .trackers import CentroidTracker, ObjectDetectionTracker
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
    # trackedObjects = {
    #   'objectID': Array of frame data
    # }
    # frameData = {
    #   'frame_id': 
    #   'other data'
    # }
    trackedObjects = {}
    def __init__(self, isolateObjectIds = []) -> None:
        super().__init__()
        self.ct = ObjectDetectionTracker()
        self.isolateObjectIds = isolateObjectIds

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        # get properties of video
        gsd = task.gsd
        fps = task.fps
        using_background_frame = task.has('background_frame')
        has_stablisation = task.has('stablisation')
        has_predictions = task.has('predictions')

        if using_background_frame:
            frame = task.get('background_frame')
        else:
            frame = task.frame
        o_width = task.frame.shape[0]
        o_height = task.frame.shape[1]
        width = frame.shape[0]
        height = frame.shape[1]

        if has_predictions:
            boxes = []
            for pred in task.get('predictions'):
                box  = pred.getBox()
                if using_background_frame:
                    x1p = box[0] / o_width
                    y1p = box[1] / o_height
                    x2p = box[2] / o_width
                    y2p = box[3] / o_height
                    box = [round(width * x1p), round(height * y1p), round(width * x2p), round(height * y2p)]
                boxes.append(box)      
            objects = self.ct.init(frame,boxes)
        else:
            objects = self.ct.update(frame)    

        # update centroid tracker with new centroids

        frame_objects = []

        # loop over the tracked objects
        for (objectID, centroid) in objects:
            if len(self.isolateObjectIds) > 0 and (objectID in self.isolateObjectIds) == False:
                continue

            if using_background_frame:
                cxp = centroid[0] / width
                xyp = centroid[1] / height
                centroid = [round(cxp * o_width), round(xyp * o_height)]

            object_dict = {}
            real_centroid = [int(centroid[0]), int(centroid[1])]
            distance_from_mid = 0
            if has_stablisation:
                sp = task.get('stablisation')
                xo = sp['centroid'][0] - real_centroid[0]
                yo = sp['centroid'][1] - real_centroid[1]
                distance_from_mid = math.sqrt(math.pow(xo,2) + math.pow(yo,2))

            if objectID in self.trackedObjects:
                # for object ID get the last frame
                last_object_frame = self.trackedObjects[objectID][-1]
                last_centroid = last_object_frame['centroid']
                last_velocity = last_object_frame['frame_velocity']

                # get time change in seconds since last time object was detected
                t = (task.frame_id - last_object_frame['frame_id']) / fps

                # velocity in pixels
                frame_velocity = [(centroid[0] - last_centroid[0]) / t, (centroid[1] - last_centroid[1]) / t]
        
                # acceleration in pixels
                frame_acceleration = [(frame_velocity[0] - last_velocity[0]) / t, (frame_velocity[1] - last_velocity[1]) / t]
                

                frame_velocity_magnitude = self.calculateMagnitudeOfVector(frame_velocity)
                frame_acceleration_magnitude = self.calculateMagnitudeOfVector(frame_acceleration)
                frame_bearing = self.calculateBearingOfVector(frame_velocity)
                frame_acceleration_bearing = self.calculateBearingOfVector(frame_acceleration)

                world_velocity = [i * gsd for i in frame_velocity]
                world_velocity_magnitude = frame_velocity_magnitude * gsd
                world_acceleration = [i * gsd for i in frame_acceleration]
                world_acceleration_magnitude = frame_acceleration_magnitude * gsd


                object_dict = {
                    'centroid': centroid,
                    'relative_centroid':real_centroid,
                    'distance_from_mid':distance_from_mid,
                    'frame_velocity': frame_velocity,
                    'frame_velocity_magnitude': frame_velocity_magnitude,
                    'frame_acceleration': frame_acceleration,
                    'frame_acceleration_magnitude': frame_acceleration_magnitude,
                    'frame_bearing': frame_bearing,
                    'frame_acceleration_bearing': frame_acceleration_bearing,
                    'world_velocity': world_velocity,
                    'world_velocity_magnitude': world_velocity_magnitude,
                    'world_acceleration': world_acceleration,
                    'world_acceleration_magnitude': world_acceleration_magnitude,
                    # world_bearing - can't implement
                    # world_acceleration_bearing - can't implement
                }
            else:
                object_dict = {
                    'centroid': centroid,
                    'relative_centroid':real_centroid,
                    'distance_from_mid':distance_from_mid,
                    'frame_velocity': [0, 0],
                    'frame_velocity_magnitude': 0,
                    'frame_acceleration': [0, 0],
                    'frame_acceleration_magnitude': 0,
                    'frame_bearing': 0,
                    'frame_acceleration_bearing': 0,
                    'world_velocity': [0, 0],
                    'world_velocity_magnitude': 0,
                    'world_acceleration': [0, 0],
                    'world_acceleration_magnitude': 0,
                }

            tracked_object = object_dict
            tracked_object['frame_id'] = task.frame_id
            if objectID in self.trackedObjects:
                self.trackedObjects[objectID].append(tracked_object)
            else:
                self.trackedObjects[objectID] = [tracked_object]
            
            object_dict['object_id'] = objectID
            frame_objects.append(object_dict)

        task.put('frame_objects', frame_objects)
        
        result = next(task)  
        #any processing after the pipeline can be done here
        return result
    def calculateBearingOfVector(self,vector):
        x = vector[0]
        y = vector[1]
        bearing = 0
        if(x > 0 and y > 0):
            # first quadrant
            bearing = np.arctan(x/y)
        elif (x > 0 and y < 0):
            # second quadrant
            bearing = (np.pi/2) + np.arctan(y/x)
        elif (x < 0 and y < 0):
            # third quadrant
            bearing = np.pi + np.arctan(x/y)
        elif (x < 0 and y > 0):
            # fourth quadrant
            bearing = ((2 * np.pi) / 3) + np.arctan(y/x)
        else:
            return 0
                    
        return np.degrees(bearing)

    def calculateMagnitudeOfVector(self,vector):
        return np.sqrt(np.power(vector[0], 2) + np.power([1], 2))[0]    

class GeoObjectTracker(PipelineHandler):
    #Data persists between frames
    # trackedGeoObjects = {
    #   'objectID': Array of frame data
    # }
    # frameData = {
    #   'frame_id': 
    #   'other data'
    # }
    trackedGeoObjects = {}
    def __init__(self) -> None:
        super().__init__()
        self.ct = CentroidTracker(isGeo=True)
    
    # called per frame
    def handle(self, task:VideoProcessingFrame, next):
        fps = task.fps

        rects = []

        geo = None
        if task.has('geo'):
            geo = task.get('geo')
        else:
            raise Exception("no geo provide")

        if task.has('predictions') and len(task.get('predictions')) > 0:
            for prediction in task.get('predictions'):
                box = np.asarray(prediction.getBox())
                rects.append(box.astype("int"))
        else:
            # return something if no predictions
            print("no predictions in frame", task.frame.frame_id)
            result = next(task)
            return result
        
        # update tracker with new centroids
        objects = self.ct.update(rects, geo=geo, task=task)
        frame_geo_objects = []

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            geo_object_dict = {}
            row, column = geo.longLatToPoint(task.frame_width, task.frame_height, centroid[0], centroid[1])

            if objectID in self.trackedGeoObjects:

                last_object_frame = self.trackedGeoObjects[objectID][-1]
                last_long_lat = last_object_frame['long_lat']
                print("this long lat", centroid[0], centroid[1])
                print("last_long_lat", last_long_lat)
                last_velocity = last_object_frame['geo_velocity']
                print("last_velocity", last_velocity)

                # get time change in seconds since last time object was detected
                t = (task.frame_id - last_object_frame['frame_id']) / fps

                # distance calculation
                forward_azimuth, backward_azimuth, distance = geo.calculateDistanceAndAzimuthBetweenTwoPoints(last_long_lat[0], last_long_lat[1], centroid[0], centroid[1])

                print("forward_azimuth, backward_azimuth, distance", forward_azimuth, backward_azimuth, distance)

                # velocity calculation
                velocity = distance / t
                print("velocity", velocity)

                # acceleration calculation
                acceleration = (velocity - last_velocity) / t
                print("acceleration", acceleration)
                
                geo_object_dict = {
                    'long_lat': centroid.tolist(),
                    'pixel_centroid': [row, column],
                    'geo_velocity': velocity,
                    'forward_azimuth': forward_azimuth,
                    'acceleration': acceleration,
                }
            else:
                geo_object_dict = {
                    'long_lat': centroid.tolist(),
                    'pixel_centroid': [row, column],
                    'geo_velocity': 0,
                    'forward_azimuth': 0,
                    'acceleration': 0,
                }

            geo_tracked_object = geo_object_dict
            geo_tracked_object['frame_id'] = task.frame_id
            if objectID in self.trackedGeoObjects:
                self.trackedGeoObjects[objectID].append(geo_tracked_object)
            else:
                self.trackedGeoObjects[objectID] = [geo_tracked_object]

            geo_object_dict['object_id'] = objectID
            frame_geo_objects.append(geo_object_dict)

        task.put('frame_geo_objects', frame_geo_objects)
        result = next(task)
        return result

class StablisationDectection(PipelineHandler):  
    def __init__(self, bbox_size = 40) -> None:
        super().__init__()
        self.bbox_size = bbox_size
        self.tracker = cv2.TrackerMIL_create()
        #x,y,w,h
        self.last_bbox = None
        self.centroids = []

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        if task.has('background_frame'):
            frame = task.get('background_frame')
        else:
            frame = task.frame

        width = frame.shape[0]
        height = frame.shape[1]
        cx = int(width * 0.5)
        cy = int(height * 0.5)

        if self.last_bbox == None:
            bbh = int(self.bbox_size / 2)
            self.last_bbox = (cx - bbh, cy - bbh, bbh*2, bbh*2)
            self.tracker.init(frame, self.last_bbox)
        else:
            success, bbox = self.tracker.update(frame)   
            if success:
                self.last_bbox = bbox 

        tbbox = self.saveBBoxForVisulisation(frame, task.frame)
        task.put('stablisation_point', tbbox)

        centroid = (tbbox[0] + (tbbox[2] / 2),tbbox[1] + (tbbox[3] / 2))
        
        movement = (0,0)
        movement_starting = (0,0)
        if len(self.centroids) > 0:
            movement = (centroid[0] - self.centroids[-1][0], centroid[1] - self.centroids[-1][1])
            movement_starting = (centroid[0] - self.centroids[0][0], centroid[1] - self.centroids[0][1])

        task.put('stablisation', {
            'centroid':centroid,
            'from_last_frame': movement,
            'from_original_frame': movement_starting
        })
        self.centroids.append(centroid)
        print('stable len', len(self.centroids))
        return next(task)     

    def saveBBoxForVisulisation(self,background_frame, original_frame):
        xp = self.last_bbox[0] / background_frame.shape[0] 
        yp = self.last_bbox[1] / background_frame.shape[1]
        w = self.bbox_size / background_frame.shape[1]
        width = original_frame.shape[0]
        height = original_frame.shape[1]
        return (width * xp, height * yp,  (width * xp) + (width * w),  (height * yp) + (height * w))

class BackgroundFrame(PipelineHandler)   :
    def __init__(self,scale = 50) -> None:
        super().__init__()
        self.scale = scale

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        img = task.frame
        width = int(img.shape[1] * self.scale / 100)
        height = int(img.shape[0] * self.scale / 100)
        dim = (width, height)   
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
        task.put('background_frame', resized)
        task.put('background_frame_scale', self.scale)
        result = next(task)   
        #cleanup frame   
        task.put('background_frame', None)
        task.put('background_frame_scale', None)
        return result

class CreateAverageImage(PipelineHandler):
    def __init__(self,save_image_path) -> None:
        super().__init__()
        self.buffer = []
        self.shape = None
        self.save_image_path = save_image_path

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        self.shape = task.frame.shape
        self.buffer.append(task.frame)   
        return next(task)   

    #called at the end                
    def after(self, processed: ProcessedVideo):
        mean_frame = np.zeros(self.shape, dtype='float32')
        for item in self.buffer:
            mean_frame += item
        mean_frame /= len(self.buffer)
        avg = mean_frame.astype('uint8')
        cv2.imwrite(self.save_image_path, avg)          

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
        result.put('output_frame', None)
        return result

    def imageWriter(self, task:VideoProcessingFrame):
        if self.image_frame_output_dir != None:
            imgPath = os.path.join(self.image_frame_output_dir, str(task.frame_id)+'_'+self.file_prefix+'.jpg')
            cv2.imwrite(imgPath, task.get('output_frame'))


    def release(self):
        if self.video != None:
            self.video.release()
        return self    

class VideoPredictionVisualisation(PipelineHandler):
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
            if self.processParam(task, 'frame_objects'):
                for object in task.get('frame_objects'):
                    centroid = object['centroid']
                    cv2.circle(output_frame, centroid, radius=5, color=self.colour, thickness=-1)
                    self.printText(output_frame, "Object ID: " + str(object['object_id']), (centroid[0]+10, centroid[1]))
                    self.printText(output_frame, "Velocity: " + str(round(object['world_velocity_magnitude'])), (centroid[0]+10, centroid[1] - 30))
                    self.printText(output_frame, "Bearing: " + str(round(object['frame_bearing'])), (centroid[0]+10, centroid[1] - 60))
                    self.printText(output_frame, "D: " + str(object['distance_from_mid']), (centroid[0]+10, centroid[1] - 90))
                    
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
                        cv2.line(output_frame, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv2.LINE_AA)   
            if self.processParam(task, 'runway_ends'):
                bboxes = task.get('runway_ends')
                for bbox in bboxes:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(output_frame, p1, p2, (255,0,0), 2, 1)                                  
            if self.processParam(task, 'frame_geo_objects'):
                for object in task.get('frame_geo_objects'):  
                    centroid = object['pixel_centroid']
                    cv2.circle(output_frame, centroid, radius=5, color=self.colour, thickness=-1)
                    self.printText(output_frame, "Object ID: " + str(object['object_id']), (centroid[0]+10, centroid[1]))
                    self.printText(output_frame, "Longitude, Latitude: " + str(round(object['long_lat'][0], ndigits=2)) + ", " + str(round(object['long_lat'][1], ndigits=2)), (centroid[0]+10, centroid[1] - 30))
                    self.printText(output_frame, "Velocity: " + str(round(object['geo_velocity'])) + "m/s", (centroid[0]+10, centroid[1] - 60))
                    self.printText(output_frame, "Azimuth: " + str(round(object['forward_azimuth'])), (centroid[0]+10, centroid[1] - 90))
            if self.processParam(task, 'stablisation_point'):
                bbox = task.get('stablisation_point')  
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(output_frame, p1, p2, (255,0,0), 2, 1)     

            task.put('output_frame', output_frame)
        return next(task)

    def processParam(self,task: VideoProcessingFrame, param):
        return (len(self.include) == 0 or param in self.include) and task.has(param) and  task.get(param) != None

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
        skip_frames = 0,
        clean_predictions_after_frame = False
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
            
    def handle(self, task: VideoProcessingFrame, next):
        print('taskid', task.frame_id)
        if task.frame_id > 0:
            return next(task)

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
from re import M
import cv2
from ..pipeline import PipelineHandler
from ..sendables import VideoProcessingFrame
from ..trackers import ObjectDetectionTracker
import numpy as np
import math    


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
    def __init__(self, isolateObjectIds = [], sanity_lines= False) -> None:
        super().__init__()
        self.ct = ObjectDetectionTracker()
        self.isolateObjectIds = isolateObjectIds
        self.sanity_lines = sanity_lines

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
            distance_from_mid = 0
            if has_stablisation and self.sanity_lines:
                sps = task.get('stablisation')
                for sp in sps:
                    xo = sp['centroid'][0] - centroid[0]
                    yo = sp['centroid'][1] - centroid[1]
                    distance_from_mid = math.sqrt(math.pow(xo,2) + math.pow(yo,2))
                    if task.has('output_frame'):
                        output_frame = task.get('output_frame')
                        #draw a santity line
                        cv2.line(
                            output_frame, 
                            [round(sp['centroid'][0]), round(sp['centroid'][1])], 
                            [round(centroid[0]), round(centroid[1])],
                            (0,255,0), 
                            2
                        )

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
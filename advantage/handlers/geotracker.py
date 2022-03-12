from ..pipeline import PipelineHandler
from ..sendables import VideoProcessingFrame
from ..trackers import CentroidTracker
import numpy as np
from yolov5.utils.augmentations import letterbox  

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
        ground_control_points = None
        if task.has('gcp'):
            ## map {longitude:0, latitude:0, x:0, y:0} x & y are top left of frame
            ground_control_points = task.get('gcp')

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

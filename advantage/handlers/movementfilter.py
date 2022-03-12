from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame
import numpy as np

class MovementFilter(PipelineHandler):
    def __init__(self) -> None:
        super().__init__()
        self.frame_buffer = 5
        self.trend_threshold = .4
        self.distances = {}
        self.trends = {}

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        filtered_frame_objects = []
        if task.has('frame_objects'):
            frameObjects = task.get('frame_objects')  
            for frameObject in frameObjects:
                objectID = frameObject['object_id']
                if (objectID in self.distances.keys()) == False:
                     self.distances[objectID] = []

                self.distances[objectID].append(frameObject['distance_from_mid'])   
                objectDistances = self.distances[objectID]  

                #idx = min(self.frame_buffer, len(self.distances[objectID] ))
                #objectDistances = self.distances[objectID][-idx:]

                if len(objectDistances) > 1:

                    #if the slope is a +ve value --> increasing trend
                    #if the slope is a -ve value --> decreasing trend
                    #if the slope is a zero value --> No trend
                    trend = self.trendDetector(objectDistances)
                    if (objectID in self.trends.keys()) == False:
                        self.trends[objectID] = []
                    elif (trend > self.trend_threshold) or (trend < -self.trend_threshold) :
                        filtered_frame_objects.append(frameObject)

                    self.trends[objectID].append(trend)

        task.put('frame_objects', filtered_frame_objects)
        return next(task)   
    
    def trendDetector(self, array_of_data, order=1):
        list_of_index = np.arange(0,len(array_of_data))
        result = np.polyfit(list_of_index, array_of_data, order)
        slope = result[-2]
        return float(slope)
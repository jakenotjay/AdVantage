from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame
import numpy as np

class MovementFilter(PipelineHandler):
    def __init__(self, trend_threshold = .4, frame_buffer=5) -> None:
        super().__init__()
        self.frame_buffer = frame_buffer
        self.trend_threshold = trend_threshold
        self.distances = {}

    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        filtered_frame_objects = []
        if task.has('frame_objects'):
            frameObjects = task.get('frame_objects')  
            for frameObject in frameObjects:
                objectID = frameObject['object_id']
                objectDistances = frameObject['stablisation_distances']

                if (objectID in self.distances.keys()) == False:
                     self.distances[objectID] = []
                     for d in objectDistances:
                        self.distances[objectID].append([]) 
                    
                passedTrendCount = 0
                for i,d in enumerate(objectDistances):
                    self.distances[objectID][i].append(d)

                    distances = self.distances[objectID][i]
                    if len(distances) > 1:
                        #if the slope is a +ve value --> increasing trend
                        #if the slope is a -ve value --> decreasing trend
                        #if the slope is a zero value --> No trend
                        trend = self.trendDetector(distances)
                        if (trend > self.trend_threshold) or (trend < -self.trend_threshold) :
                            passedTrendCount += 1
                if passedTrendCount == len(objectDistances):
                    filtered_frame_objects.append(frameObject)

        task.put('frame_objects', filtered_frame_objects)
        return next(task)   
    
    def trendDetector(self, array_of_data, order=1):
        list_of_index = np.arange(0,len(array_of_data))
        result = np.polyfit(list_of_index, array_of_data, order)
        slope = result[-2]
        return float(slope)
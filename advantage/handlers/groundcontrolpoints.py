from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame

class GroundControlPoints(PipelineHandler):
     def __init__(self) -> None:
         self.points = [
             {
                 'longitude':174.7700001,
                 'latitude':-37.0138851,
                 'x':80,
                 'y':2350,
                 'size':100
             },
             {
                 'longitude':174.8008117,
                 'latitude':-37.007909,
                 'x':2515,
                 'y':1190,
                 'size':100
             }
         ]
     def handle(self, task: VideoProcessingFrame, next):
         task.put('gcp', self.points)
         return next(task)
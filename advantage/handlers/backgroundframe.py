from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame
import cv2

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
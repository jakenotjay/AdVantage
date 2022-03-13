from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame
from ..lib import ProcessedVideo
import cv2
import numpy as np

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
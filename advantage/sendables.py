from cv2 import VideoCapture
from .pipeline import PipelineObject


class VideoProcessingFrame(PipelineObject):
    video = None
    frame = None
    frame_id = 0
    def __init__(self, video: VideoCapture, frame, frame_id) -> None:
        super().__init__()
        self.video = video
        self.frame = frame
        self.frame_id = frame_id


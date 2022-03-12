from cv2 import VideoCapture
from .pipeline import PipelineObject


class VideoProcessingFrame(PipelineObject):
    video = None
    frame = None
    frame_id = 0
    frame_width = 0
    frame_height = 0
    fps = 0
    gsd = 1.2 # hard coded sorry Carl :(
    continue_frames = True
    def __init__(self, video: VideoCapture, frame, frame_id, frame_width, frame_height, fps) -> None:
        super().__init__()
        self.video = video
        self.frame = frame
        self.frame_id = frame_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.continue_frames = True

    def clean(self):
        self.frame = None

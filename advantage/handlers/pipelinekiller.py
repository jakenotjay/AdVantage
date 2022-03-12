from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame

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
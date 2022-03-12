from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame

class FrameBuffer(PipelineHandler):
    buffer = []
    buffer_size = 0
    def __init__(self, buffer_size = 3) -> None:
        super().__init__()
        self.buffer_size = buffer_size

    def handle(self, task: VideoProcessingFrame, next):
        self.buffer.append(task)
        if len(self.buffer) > self.buffer_size: 
            self.buffer = self.buffer[-(self.buffer_size):]
        task.put('frame_buffer', self.buffer)
        return next(task)   
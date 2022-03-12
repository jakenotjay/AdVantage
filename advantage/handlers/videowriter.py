from ..pipeline import PipelineHandler
from ..app import VideoProcessingFrame
import os
import cv2

class VideoWriter(PipelineHandler):
    video = None
    outputFile = None
    image_frame_output_dir = None
    file_prefix = ''
    output_video = True
    def __init__(self, output_file_name, output_video = True, image_frame_output_dir = None) -> None:
        super().__init__()
        self.outputFile = output_file_name
        self.image_frame_output_dir = image_frame_output_dir
        self.file_prefix = ''
        self.output_video = output_video
        if image_frame_output_dir != None:
            self.file_prefix = os.path.splitext(os.path.basename(output_file_name))[0]


    def handle(self, task: VideoProcessingFrame, next):
        if self.video == None and self.output_video:
            self.video = cv2.VideoWriter(self.outputFile, cv2.VideoWriter_fourcc(*'DIVX'), task.fps, (task.frame_width, task.frame_height)) 
        task.put('output_frame', task.frame.copy())
        result = next(task)
        if self.output_video:
            self.video.write(result.get('output_frame'))
        self.imageWriter(result)
        result.put('output_frame', None)
        return result

    def imageWriter(self, task:VideoProcessingFrame):
        if self.image_frame_output_dir != None:
            imgPath = os.path.join(self.image_frame_output_dir, str(task.frame_id)+'_'+self.file_prefix+'.jpg')
            cv2.imwrite(imgPath, task.get('output_frame'))


    def release(self):
        if self.video != None:
            self.video.release()
        return self    
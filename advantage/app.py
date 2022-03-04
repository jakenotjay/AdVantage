import cv2
from .pipeline import Pipeline
from .sendables import VideoProcessingFrame
from .lib import ProcessedVideo
from .pipeline import Pipeline

class AdVantage:

    def pipeline_factory(tasks):
        return Pipeline(tasks)

    def process_video(videoFile, pipeline: Pipeline):
        processedVideo = ProcessedVideo()
        video = cv2.VideoCapture(videoFile)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_id = -1
        while True:
            frame_id += 1
            ret, frame = video.read()
            if not ret:
                break 
            process = VideoProcessingFrame(video, frame, frame_id, frame_width, frame_height, fps)
            processedVideo.append(pipeline.send(process))
        
        return processedVideo


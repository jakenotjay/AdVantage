import cv2
from .pipeline import Pipeline
from .sendables import VideoProcessingFrame
from .lib import ProcessedVideo
from .pipeline import Pipeline
import json

class AdVantage:

    def pipeline_factory(self,tasks):
        return Pipeline(tasks)

    def process_video(self, videoFile, pipeline: Pipeline):
        processedVideo = ProcessedVideo()
        video = cv2.VideoCapture(videoFile)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_id = -1
        pipeline.reverseHandlers()
        while True:
            frame_id += 1
            ret, frame = video.read()
            if not ret:
                break 
            process = VideoProcessingFrame(video, frame, frame_id, frame_width, frame_height, fps)
            processedFrame = pipeline.send(process)
            processedVideo.append(processedFrame)
        
            if processedFrame.continue_frames == False:
                break
          
        return processedVideo

    def save_output_to_json(self, processed_video: ProcessedVideo, output_file):
        outputList = []
        for frame in processed_video.getFrames():
            frameMap = {
                'frame_id': frame.frame_id
            }
            if frame.has('predictions'):
                frameMap['predictions'] = []
                for pred in frame.get('predictions'):
                    frameMap['predictions'].append(pred.toMap())

            outputList.append(frameMap)

        open(output_file,"w").write(json.dumps(outputList))    

            


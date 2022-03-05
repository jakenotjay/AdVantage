import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *

app = AdVantage()
cwd = os.path.dirname(os.path.abspath(__file__))


pipeline = app.pipeline_factory([
    #Verbose(), #Prints information to console
    #PipelineKiller(frames_to_process = 2), #kills process after x frames
    #FrameBuffer(buffer_size = 2), #Keep current and last x frames
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'), #Attach geo data to frame
    VideoWriter(
        os.path.join(cwd,'output','VX020001c0_stable.mp4'), 
        output_video=True, 
        #image_frame_output_dir=os.path.join(cwd,'output')
    ),
    YoloProcessor(os.path.join(cwd,'input/best.pt'),conf_thres=0.7), 
    VideoPredictionVisulisation(),
])

result = app.process_video(os.path.join(cwd,'input','VX020001c0_stable.mp4'), pipeline)
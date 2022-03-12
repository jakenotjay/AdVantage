import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *
from advantage.runwaydetector import RunwayDetector

app = AdVantage()
cwd = os.path.dirname(os.path.abspath(__file__))
#filename = 'VX020001dc'
#filename = 'VX0200021f'
filename = 'VX020001c0'
#filename = 'VX020001c0_stable'


pipeline = app.pipeline_factory([
    #Verbose(), #Prints information to console
    #PipelineKiller(frames_to_process = 10), #kills process after x frames
    #FrameBuffer(buffer_size = 2), #Keep current and last x frames
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'), #Attach geo data to frame
    BackgroundFrame(scale=80), #Creates a resized frame to process on
    StablisationDectection(bbox_size=40),
    VideoWriter(
        os.path.join(acwd,'output',filename+'.mp4'), #Video Path to Save to if set to true
        output_video=True, #Output the video
        #image_frame_output_dir=os.path.join(cwd,'output') #Outputs Image of each frame
    ),
    YoloProcessor(
       os.path.join(cwd,'input/exp3/best.pt'), #weights file. must be absolute
       conf_thres=0.7, #only save predictions over % 0 to 1
       clean_predictions_after_frame=True
    ), 
    ObjectTracker(isolateObjectIds=[1,9]),
    #RunwayDetector(output_test_images=False),
    VideoPredictionVisualisation(include=['frame_objects','stablisation_point']), # Applies details to video/image frames
])

result = app.process_video(os.path.join(cwd,'input',filename+'.mp4'), pipeline)
app.save_output_to_json(result, os.path.join(cwd,'output',filename+'.json'))
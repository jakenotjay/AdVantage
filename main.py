import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *
from advantage.handlers.objectracker import ObjectTracker
from advantage.handlers.verbose import Verbose
from advantage.handlers.groundcontrolpoints import GroundControlPoints
from advantage.handlers.backgroundframe import BackgroundFrame
from advantage.handlers.stablisationdectection import StablisationDetection
from advantage.handlers.videowriter import VideoWriter
from advantage.handlers.yolo import YoloProcessor
from advantage.handlers.movementfilter import MovementFilter
from advantage.handlers.visulisation import VideoPredictionVisualisation

app = AdVantage()
cwd = os.path.dirname(os.path.abspath(__file__))
#filename = 'VX020001dc'
#filename = 'VX0200021f'
filename = 'VX020001c0'
#filename = 'VX020001c0_stable'


pipeline = app.pipeline_factory([
    Verbose(), #Prints information to console
    #PipelineKiller(frames_to_process = 1), #kills process after x frames
    GroundControlPoints(), #hard coded for VX020001c0 frame 1
    BackgroundFrame(scale=80), #Creates a resized frame to process on
    StablisationDetection(bbox_size=100),
    VideoWriter(
        os.path.join(cwd,'output',filename+'.mp4'), #Video Path to Save to if set to true
        output_video=True, #Output the video
        #image_frame_output_dir=os.path.join(cwd,'output') #Outputs Image of each frame
    ),
    YoloProcessor(
       os.path.join(cwd,'input/exp3/best.pt'), #weights file. must be absolute
       conf_thres=0.7, #only save predictions over % 0 to 1
    ), 
    ObjectTracker(isolateObjectIds=[], sanity_lines=True), #nice ones to check 9,8,11,1
    MovementFilter(),
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'), #Attach geo data to frame
    #RunwayDetector(output_test_images=False),
    VideoPredictionVisualisation(include=['frame_objects','stablisation_points']), # Applies details to video/image frames
])

result = app.process_video(os.path.join(cwd,'input',filename+'.mp4'), pipeline)
app.save_output_to_json(result, os.path.join(cwd,'output',filename+'.json'))
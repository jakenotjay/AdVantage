import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *

app = AdVantage
cwd = os.path.dirname(os.path.abspath(__file__))

pipeline = app.pipeline_factory([
    Verbose(),
    VideoWriter('output/VX020003b9.mp4'),
    YoloProcessor(os.path.join(cwd,'input/best.pt')), 
    VideoPredictionVisulisation(),
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'),
])

result = app.process_video('input/VX020003b9.mp4', pipeline)
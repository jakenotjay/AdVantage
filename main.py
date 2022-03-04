import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *

app = AdVantage
cwd = os.path.dirname(os.path.abspath(__file__))

videos = ['VX020001af','VX02000399','VX020003b5','VX020001c0','VX0200021f','VX020001dc','VX020003a0','VX020003b9']


pipeline = app.pipeline_factory([
    Verbose(),
    FrameBuffer(buffer_size = 2),
    VideoWriter('output/VX020001c0.mp4'),
    YoloProcessor(os.path.join(cwd,'input/best.pt'),conf_thres=0.7), 
    VideoPredictionVisulisation(),
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'),
])

result = app.process_video('input/VX020001c0.mp4', pipeline)
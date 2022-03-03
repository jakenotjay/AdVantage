import os
import sys
sys.path.insert(0, './yolov5')

from advantage.app import AdVantage
from advantage.handlers import *

app = AdVantage
cwd = os.path.dirname(os.path.abspath(__file__))



pipeline = app.pipeline_factory([
    Verbose(),
    YoloProcessor(os.path.join(cwd,'input/best.pt')),
    #VideoAttachGeoData('input/VX020001c0_geometry.xml'),
    #VideoWriter('output/VX020001c0.mp4',3072,3072,10)
])

result = app.process_video('input/VX020001c0.mp4', pipeline)
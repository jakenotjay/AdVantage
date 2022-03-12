from re import M
import cv2
from ..pipeline import PipelineHandler
from ..sendables import VideoProcessingFrame

class VideoPredictionVisualisation(PipelineHandler):
    colour = None
    size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX,
    fontScale = 1,
    fontColour = (0, 0, 255),
    fontThickness = 2
    include = []

    def __init__(
        self,
        colour=(255, 0, 0),
        size= 2,
        font = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 1,
        fontColour = (0, 0, 255),
        fontThickness = 2,
        include = []
    ) -> None:
        super().__init__()
        self.colour = colour
        self.size = size
        self.font = font
        self.fontScale = fontScale
        self.fontColour = fontColour
        self.fontThickness = fontThickness
        self.include = include

    def handle(self, task: VideoProcessingFrame, next):
        if task.has('output_frame'):
            output_frame = task.get('output_frame')
            if self.processParam(task, 'frame_objects'):
                for object in task.get('frame_objects'):
                    centroid = object['centroid']
                    cv2.circle(output_frame, centroid, radius=5, color=self.colour, thickness=-1)
                    self.printText(output_frame, "Object ID: " + str(object['object_id']), (centroid[0]+10, centroid[1]))
                    self.printText(output_frame, "Velocity: " + str(round(object['world_velocity_magnitude'])), (centroid[0]+10, centroid[1] - 30))
                    self.printText(output_frame, "Bearing: " + str(round(object['frame_bearing'])), (centroid[0]+10, centroid[1] - 60))
                    #self.printText(output_frame, "D: " + str(object['distance_from_mid']), (centroid[0]+10, centroid[1] - 90))
                    
            if self.processParam(task, 'predictions'):
                for prediction in task.get('predictions'):
                    box = prediction.getBox()
                    cv2.rectangle(output_frame, (box[0],box[1]),(box[2],box[3]), self.colour, self.size)
                    self.printText(output_frame, prediction.getLabel() , (box[2] - 50,box[3] + 50))
                    self.printText(output_frame, str(prediction.getScore()) , (box[2] - 100,box[3] + 100))
            if self.processParam(task, 'runways'):
                lines = task.get('runways')
                if lines is not None:
                    for i in range(0, len(lines)):
                        l = lines[i]
                        cv2.line(output_frame, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2, cv2.LINE_AA)   
            if self.processParam(task, 'runway_ends'):
                bboxes = task.get('runway_ends')
                for bbox in bboxes:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[2]), int(bbox[3]))
                    cv2.rectangle(output_frame, p1, p2, (255,0,0), 2, 1)                                  
            if self.processParam(task, 'frame_geo_objects'):
                for object in task.get('frame_geo_objects'):  
                    centroid = object['pixel_centroid']
                    cv2.circle(output_frame, centroid, radius=5, color=self.colour, thickness=-1)
                    self.printText(output_frame, "Object ID: " + str(object['object_id']), (centroid[0]+10, centroid[1]))
                    self.printText(output_frame, "Longitude, Latitude: " + str(round(object['long_lat'][0], ndigits=2)) + ", " + str(round(object['long_lat'][1], ndigits=2)), (centroid[0]+10, centroid[1] - 30))
                    self.printText(output_frame, "Velocity: " + str(round(object['geo_velocity'])) + "m/s", (centroid[0]+10, centroid[1] - 60))
                    self.printText(output_frame, "Azimuth: " + str(round(object['forward_azimuth'])), (centroid[0]+10, centroid[1] - 90))
            if self.processParam(task, 'stablisation_points'):
                points = task.get('stablisation_points')  
                for bbox in points:
                    p1 = (round(bbox[0]), round(bbox[1]))
                    p2 = (round(bbox[2]), round(bbox[3]))
                    cx = round(bbox[0] + ((bbox[2] - bbox[0]) / 2))
                    cy = round(bbox[1] + ((bbox[3] - bbox[1]) / 2))
                    centroid = [cx, cy]
                    cv2.circle(output_frame, centroid, radius=10, color=self.fontColour, thickness=-1)
                    cv2.rectangle(output_frame, p1, p2, (255,0,0), 2, 1)     

            task.put('output_frame', output_frame)
        return next(task)

    def processParam(self,task: VideoProcessingFrame, param):
        return (len(self.include) == 0 or param in self.include) and task.has(param) and  task.get(param) != None

    def printText(self,frame, text, position):
        cv2.putText(
            frame, 
            text, 
            position, 
            self.font, 
            self.fontScale, 
            self.fontColour, 
            self.fontThickness, 
            cv2.LINE_AA
        ) 

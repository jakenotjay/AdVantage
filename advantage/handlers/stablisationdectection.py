from re import M
import cv2
from ..pipeline import PipelineHandler
from ..sendables import VideoProcessingFrame


class StablisationDetection(PipelineHandler):  
    def __init__(self, bbox_size = 40) -> None:
        super().__init__()
        self.bbox_size = bbox_size
        self.trackers = []
        #x,y,w,h
        self.last_bbox = None
        self.x_offset = .5
        self.y_offset = .5
        self.centroids = []

    def setupTracker(self, coords, size, frame, o_frame):
        tracker = cv2.TrackerMIL_create()
        xp = coords[0] / o_frame.shape[0]
        yp = coords[1] / o_frame.shape[1]
        sp = size / o_frame.shape[0]

        s = round(frame.shape[0] * sp)
        x = round(frame.shape[0] * xp)
        y = round(frame.shape[1] * yp)
        tracker.init(frame, [x,y,s,s])
        self.trackers.append({
            'tracker':tracker,
            'box':[x,y,s,s]
        })


    #Called Per Frame
    def handle(self, task: VideoProcessingFrame, next):
        if task.has('background_frame'):
            frame = task.get('background_frame')
        else:
            frame = task.frame  

        if task.frame_id == 0:      
            if task.has('gcp'):
                for point in task.get('gcp'):
                    self.setupTracker([point['x'],point['y']], point['size'], frame, task.frame)
                    self.centroids.append([])
            else:
                cx = task.frame.shape[0] * self.x_offset
                cy = task.frame.shape[1] * self.y_offset      
                self.setupTracker([cx, cy], self.bbox_size, frame, task.frame)
                self.centroids.append([])
        else:
            for t in self.trackers:
                _, bbox = t['tracker'].update(frame)
                t['box'] = bbox

        stable_points = []
        frame_points = []
        for i,t in enumerate(self.trackers):
            tbbox = self.saveBBoxForVisulisation(frame, task.frame, t['box'])
            stable_points.append(tbbox)
            cx = tbbox[0] + ((tbbox[2] - tbbox[0])/2)
            cy = tbbox[1] + ((tbbox[3] - tbbox[1])/2)
            centroid = (cx, cy)
            movement = (0,0)
            movement_starting = (0,0)
            if len(self.centroids[i]) > 0:
                movement = (centroid[0] - self.centroids[i][-1][0], centroid[1] - self.centroids[i][-1][1])
                movement_starting = (centroid[0] - self.centroids[i][0][0], centroid[1] - self.centroids[i][0][1])

            frame_points.append({
                'centroid':centroid,
                'from_last_frame': movement,
                'from_original_frame': movement_starting
            })    

            self.centroids[i].append(centroid)

        task.put('stablisation',frame_points)    
        task.put('stablisation_points', stable_points)

        return next(task)     

    def saveBBoxForVisulisation(self,background_frame, original_frame, box):
        xp = box[0] / background_frame.shape[0] 
        yp = box[1] / background_frame.shape[1]
        w = box[2] / background_frame.shape[1]
        width = original_frame.shape[0]
        height = original_frame.shape[1]
        return (width * xp, height * yp,  (width * xp) + (width * w),  (height * yp) + (height * w))

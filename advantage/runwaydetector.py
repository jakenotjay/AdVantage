from .pipeline import PipelineHandler
from .app import VideoProcessingFrame
import cv2
import math
import numpy as np

class RunwayDetector(PipelineHandler):
    image_width = None
    output_test_images = False
    trackers = []
    def __init__(self, image_width=480, output_test_images = False, detect_stablisation_offsets = False) -> None:
        super().__init__()
        self.lines = []
        self.image_width = image_width
        self.output_test_images = output_test_images
        #self.tracker = cv2.TrackerMIL_create()
        self.trackBox = None
        self.trackers = []
        self.detect_stablisation_offsets = detect_stablisation_offsets

    def handle(self, task: VideoProcessingFrame, next):

        resize_img_width = self.image_width if self.image_width != None else task.frame_width
        img = task.frame.copy()
        scale = resize_img_width / img.shape[1]
        width = int(img.shape[1] * scale)
        height = int(img.shape[0] * scale)
        dim = (width, height)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        offset_y = [0,0,0,0]
        offset_x = [0,0,0,0]
        img_copy = img.copy()
        if self.detect_stablisation_offsets:
            offset_x, offset_y = self.findStablisedOffset(img)

        bin_img = self.convertImageToStandardisedBinary(img)
        lines = self.findLinesInBinaryImage(bin_img, width, offset_x, offset_y)
        lines = self.saveLinesToTask(task,img_copy, lines, width, height)
       
        self.addNewTrackers(task.frame,lines)
        bboxes = self.updateTrackers(task.frame)

        img_copy, bin_img = self.saveTestImages(task, img_copy, bin_img)

        task.put('runway_ends', bboxes)
        
        return next(task)

    def updateTrackers(self, frame):
        bboxes = []
        for idx, tracker in enumerate(self.trackers):
            ok, bbox = tracker['tracker'].update(frame)
            self.trackers[idx]['centroid'] = bbox

            bboxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3]
            ])

            #if self.output_test_images:
            #    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1] + bbox[3]), (0,255,0),1, cv2.LINE_AA)
            #x1p = bbox[0] / width 
            #x2p = (bbox[2] + bbox[0]) / width
            #y1p = bbox[1] / height
            #y2p = (bbox[1] + bbox[3]) / height
            #bboxes.append([
            #        int(task.frame_width * x1p),
            #        int(task.frame_height * y1p),
            #        int(task.frame_width * x2p),
            #        int(task.frame_height * y2p),
            #    ])
        return bboxes    

    def addNewTrackers(self,frame, lines, pixel_jitter = 100):
        for line in lines:
            addTracker = True
            for tracker in self.trackers:
                centroid = tracker['centroid']
                cx = centroid[0] + (centroid[2] / 2)
                cy = centroid[1] + (centroid[3] / 2)
                d = math.sqrt(math.pow(abs(cx - line[0]),2) + math.pow(abs(cy - line[1]),2))
                if d <= pixel_jitter:
                    addTracker = False
                    break

            if addTracker:
                tracker = cv2.TrackerMIL_create()
                bbox = self.createBBoxForTracker(line)
                tracker.init(frame, bbox)
                self.trackers.append({
                    'centroid': bbox,
                    'tracker': tracker
                })    


    def createBBoxForTracker(self, centroid, size_half=50):
        return (
            max(centroid[0] - size_half, 0),
            max(centroid[1] - size_half, 0),
            size_half*2,
            size_half*2
        )

    def saveTestImages(self, task, img, bin_img):
        if self.output_test_images:
            cv2.imwrite('output/frame_'+str(task.frame_id)+'_thresh.jpg', bin_img)
            cv2.imwrite('output/frame_'+str(task.frame_id)+'.jpg', img)   
        return img, bin_img    

    def saveLinesToTask(self,task,img, lines, width, height):
        outputLines = []
        if lines is not None:
            lines = sorted(lines, key=self.getLineLength)
            for i in range(0, len(lines)):
                l = lines[i]
                x1p = l[0] / width 
                x2p = l[2] / width
                y1p = l[1] / height
                y2p = l[3] / height
                outputLines.append([
                    int(task.frame_width * x1p),
                    int(task.frame_height * y1p),
                    int(task.frame_width * x2p),
                    int(task.frame_height * y2p),
                ])
                if self.output_test_images:
                    cv2.line(img, (l[0], l[1]), (l[2], l[3]), (0,0,255), 2, cv2.LINE_AA)  

            task.put('runways', outputLines)
        return outputLines    

    def findLinesInBinaryImage(self, img, width, offset_x, offset_y, edge_offset=10):
        runway_length_drop_percentage = 10
        currentRunwayDropPercentage = runway_length_drop_percentage
        currentRunwayMin = width - (width * (currentRunwayDropPercentage / 100))
        lines = None

        new_x = offset_y[0] + edge_offset
        new_y = offset_x[0] + edge_offset

        roi = img[new_x:img.shape[1] - offset_y[2] - edge_offset, new_y:img.shape[0] - offset_x[2] - edge_offset]

        while currentRunwayDropPercentage < 100:
            lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 20, None, currentRunwayMin, 10)
            if lines is not None:
                break  
            currentRunwayDropPercentage += runway_length_drop_percentage
            currentRunwayMin = width - (width * (currentRunwayDropPercentage / 100))
        new_lines = []
        for i in range(0, len(lines)):
            l = lines[i][0]
            new_lines.append([
                l[0] + new_y,
                l[1] + new_x,
                l[2] + new_y,
                l[3] + new_x
            ])

        return new_lines  

    def findStablisedOffset(self, img):
        check_points = 2
        check_x = int(img.shape[1] / (check_points + 1))
        check_y = int(img.shape[0] / (check_points + 1))
        offset_y = [0,0,0,0]
        offset_x = [0,0,0,0]
        
        broken = False
        for y in range(img.shape[0]):
            for p in range(check_points):
                x = (p+1) * check_x
                if img[x,y][0] == 0 and img[x,y][1] == 0:
                    offset_x[p] += 1
                else:
                    broken = True
                    break 
            if broken:
                break    
        
        broken = False
        for x in range(img.shape[1]):
            for p in range(check_points):
                y = (p+1) * check_y
                if img[x,y][0] == 0 and img[x,y][1] == 0:
                    offset_y[p] += 1   
                else:
                    broken = True
                    break  
            if broken:
                break  

        broken = False
        for y in range(img.shape[0]):
            y2 = img.shape[0] - (y+1) 
            for p in range(check_points):
                x = ((p+1) * check_x)
                if img[x,y2][0] == 0 and img[x,y2][1] == 0:
                    offset_x[3 - p] += 1
                else:
                    broken = True
                    break 
            if broken:
                break    
        
        broken = False
        for x in range(img.shape[1]):
            x2 = img.shape[1] - (x +1)
            for p in range(check_points):
                y = ((p+1) * check_y)
                if img[x2,y][0] == 0 and img[x2,y][1] == 0:
                    offset_y[3 - p] += 1   
                else:
                    broken = True
                    break  
            if broken:
                break          
                         
        return [offset_x, offset_y]


    def convertImageToStandardisedBinary(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.bitwise_not(img_gray) #invert image
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img_gray = cv2.dilate(img_gray, rect_kernel, iterations = 1)
        thresh = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,55,4)
        white_count = cv2.countNonZero(thresh)
        black_count = (img.shape[0] * img.shape[1]) - white_count
        if white_count > black_count:
            thresh = cv2.bitwise_not(thresh)
        return thresh
        

    def getLineLength(self, line):
        l = line
        x = l[2] - l[0]
        y = l[3] - l[1]
        return math.sqrt(math.pow(x,2) * math.pow(y, 2))    
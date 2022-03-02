import xml.etree.ElementTree as ET
import geopy.distance
import math
import numpy as np

NAMESPACE = '{http://www.spacemetric.com/}'

class VantageFrame:
    frame = None
    cache = {}

    def __init__(self, frame):
        self.frame = frame
        self.cache = {}

    def getFrameId(self):
        if 'frameId' not in self.cache:
            self.cache['frameId'] = self.frame.find(NAMESPACE+'frameId').text
        return self.cache['frameId']   

    def getSceneCentreLongLat(self):
        if 'sceneCentreLongLat' not in self.cache:
            text = self.frame.find(NAMESPACE+'sceneCentreLongLat').text.split(' ')
            self.cache['sceneCentreLongLat'] = [float(text[0]), float(text[1])]
        return self.cache['sceneCentreLongLat']    

    # [[long,lat]...]
    def getSceneFrameLongLat(self):
        if 'sceneFrameLongLat' not in self.cache:
            points = []
            textPoints = self.frame.find(NAMESPACE+'sceneFrameLongLat').text.split(',')
            for p in textPoints:
                gotPoint = p.split(' ')
                points.append([float(gotPoint[0]), float(gotPoint[1])])     
            _removed_element = points.pop()
            points.reverse()
    
            self.cache['sceneFrameLongLat'] = points
        return self.cache['sceneFrameLongLat']      

    def getViewingAngle(self):
        if 'viewingAngle' not in self.cache:
            self.cache['viewingAngle'] = float(self.frame.find(NAMESPACE+'viewingAngle').text)
        return self.cache['viewingAngle']

    def getGSD(self):
        if 'gsd' not in self.cache:
            self.cache['gsd'] = float(self.frame.find(NAMESPACE+'gsd').text)
        return self.cache['gsd']    

    def getViewingAngleAlongTrack(self):
        if 'viewingAngleAlongTrack' not in self.cache:
            self.cache['viewingAngleAlongTrack'] = float(self.frame.find(NAMESPACE+'viewingAngleAlongTrack').text)
        return self.cache['viewingAngleAlongTrack']

    def getViewingAngleAcrossTrack(self):
        if 'viewingAngleAcrossTrack' not in self.cache:
            self.cache['viewingAngleAcrossTrack'] = float(self.frame.find(NAMESPACE+'viewingAngleAcrossTrack').text)
        return self.cache['viewingAngleAcrossTrack'] 

    def getSatelliteAzimuthAngle(self):
        if 'satelliteAzimuthAngle' not in self.cache:
            self.cache['satelliteAzimuthAngle'] = float(self.frame.find(NAMESPACE+'satelliteAzimuthAngle').text)
        return self.cache['satelliteAzimuthAngle'] 

    def getSatelliteElevationAngle(self):
        if 'satelliteElevationAngle' not in self.cache:
            self.cache['satelliteElevationAngle'] = float(self.frame.find(NAMESPACE+'satelliteElevationAngle').text)
        return self.cache['satelliteElevationAngle'] 

    def getIncidenceAngle(self):
        if 'incidenceAngle' not in self.cache:
            self.cache['incidenceAngle'] = float(self.frame.find(NAMESPACE+'incidenceAngle').text)
        return self.cache['incidenceAngle']     

    def getSatelliteLongLat(self):
        if 'satelliteLongLat' not in self.cache:
             text = self.frame.find(NAMESPACE+'satelliteLongLat').text.split(' ')
             self.cache['satelliteLongLat'] = [float(text[0]), float(text[1])]  
        return self.cache['satelliteLongLat'] 
          
    def getSatelliteAltitude(self):
        if 'satelliteAltitude' not in self.cache:
            self.cache['satelliteAltitude'] = float(self.frame.find(NAMESPACE+'satelliteAltitude').text)
        return self.cache['satelliteAltitude']   

    def getRange(self):
        if 'range' not in self.cache:
            self.cache['range'] = float(self.frame.find(NAMESPACE+'range').text)
        return self.cache['range'] 

    def getFrameLongDistanceInMeters(self):
        longLatFrame = self.getSceneFrameLongLat()      
        coords_1 = (longLatFrame[0][1], longLatFrame[0][0])
        coords_2 = (longLatFrame[3][1], longLatFrame[3][0])
        return geopy.distance.geodesic(coords_1, coords_2).meters 

    def getFrameLatDistanceInMeters(self):
        longLatFrame = self.getSceneFrameLongLat()      
        coords_1 = (longLatFrame[0][1], longLatFrame[0][0])
        coords_2 = (longLatFrame[2][1], longLatFrame[2][0])
        return geopy.distance.geodesic(coords_1, coords_2).meters     

    def getLongLatDifferenceFromFrame(self, frame):
        longLatFrame = self.getSceneFrameLongLat()
        longLatFrame2 = frame.getSceneFrameLongLat()
        topLeft = longLatFrame[0]
        topLeft2 = longLatFrame2[0]
        return [topLeft2[0] - topLeft[0],topLeft2[1] - topLeft[1]] 

    def getDistanceFromFrame(self, frame):
        longLatFrame = self.getSceneFrameLongLat()
        longLatFrame2 = frame.getSceneFrameLongLat()
        coords_1 = (longLatFrame[0][1], longLatFrame[0][0])
        coords_2 = (longLatFrame2[0][1], longLatFrame2[0][0])
        distance = geopy.distance.geodesic(coords_1, coords_2).meters 
        return distance   

    def isSouthernHem(self):
        center = self.getSceneCentreLongLat()
        return center[1] <= 0.0


    def pointToLongLat(self, image_x, image_y, point_x, point_y):
        box = self.getSceneFrameLongLat()

        xip = (point_x / image_x)
        yip = (point_y / image_y)

        R = 6378.1 * 1000
        my = (point_y * self.getGSD())
        mx = (point_x * self.getGSD())
        d = math.sqrt(math.pow(my, 2) + math.pow(mx, 2))
        brng = np.arcsin(my / d) + (math.pi / 2)
       
        lon1 = math.radians(box[0][0])
        lat1 = math.radians(box[0][1])
        
        print('image', point_y, point_x, image_y, image_x)
        print('mx', mx,'my', my,'d',d)
        print('bearing', brng)

        lat2 = math.asin( 
            math.sin(lat1)*math.cos(d/R) +
            math.cos(lat1)*math.sin(d/R)*math.cos(brng)
        )

        lon2 = lon1 + math.atan2(
            math.sin(brng)*math.sin(d/R)*math.cos(lat1),
            math.cos(d/R)-math.sin(lat1)*math.sin(lat2)
        )

        lat2 = math.degrees(lat2)
        lon2 = math.degrees(lon2)

        return [lon2, lat2]


                              

class VantageGeometry:
    filename = None
    root = None
    frames = []

    def __init__(self, fileName):
        self.filename = fileName
        self.root = ET.parse(fileName).getroot()
        self.frame = []

    def getFrames(self):
        if len(self.frames) == 0:
            for scene in self.root.findall(NAMESPACE+'scene'):
                for frame in scene.findall(NAMESPACE+'frame'):
                    self.frames.append(VantageFrame(frame))
        return self.frames 

    def getFrame(self, frameId):
        frames = self.getFrames()
        if len(frames) >= frameId:
            return frames[frameId]    
        return None       

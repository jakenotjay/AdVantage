import xml.etree.ElementTree as ET
import geopy.distance
import math
import numpy as np
from pyproj.transformer import Transformer, CRS
from rasterio.transform import from_bounds

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
        # [[long, lat], ...]
        box = self.getSceneFrameLongLat()

        longs = [point[0] for point in box]
        lats = [point[1] for point in box]

        # is image_x, and image_y width and height?
        width, height = image_x, image_y 

        north = max(lats)
        west = min(longs)
        east = max(longs)
        south = min(lats)

        # flat wgs84 in metres
        crs = CRS.from_epsg(3857)

        # projection transform, always_xy = True as axes are reversed for 3857
        proj = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)

        west, north = proj.transform(west, north)
        east, south = proj.transform(east, south)

        # create affine transform matrix
        transform = from_bounds(west, south, east, north, width, height)

        # matrix multiplication of transform and row, column of image to get 
        # position in flat projection
        easting, northing = transform * (point_x, point_y)

        # reverse projection back to geodetic crs
        revProj = Transformer.from_crs(crs, crs.geodetic_crs, always_xy=True)

        # convert flat projection back to geodetic
        point_long, point_lat = revProj.transform(easting, northing)

        return point_long, point_lat   

    def longLatToPoint(self, image_x, image_y, long, lat):
        # [[long, lat], ...]
        box = self.getSceneFrameLongLat()

        longs = [point[0] for point in box]
        lats = [point[1] for point in box]
        width, height = image_x, image_y 

        north = max(lats)
        west = min(longs)
        east = max(longs)
        south = min(lats)

        # flat wgs84 in metres
        crs = CRS.from_epsg(3857)

        # projection transform, always_xy = True as axes are reversed for 3857
        proj = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)

        west, north = proj.transform(west, north)
        east, south = proj.transform(east, south)

        # create affine transform matrix for row, column to lat long
        transform = from_bounds(west, south, east, north, width, height)

        # inverse the transform to get matrix for lat, long to row,column
        inverse_transform = transform.__invert__()
        # convert lat long to easting and northing (flat surface)
        point_easting, point_northing = proj.transform(long, lat)

        row, column = inverse_transform * (point_easting, point_northing)

        return int(row), int(column)

    def calculateDistanceBetweenTwoPoints(self, long1, lat1, long2, lat2):
        # get geodetic shape
        geod_wgs84 = CRS("epsg:4326").get_geod()
        longs = [long1, long2]
        lats = [lat1, lat2]

        return geod_wgs84.line_length(longs, lats)

    def calculateDistanceAndAzimuthBetweenTwoPoints(self, long1, lat1, long2, lat2):
        geod_wgs84 = CRS("epsg:4326").get_geod()
        # determines forward and backwards azimuth (bearing) as well as distance in metres
        # forward = point1 to point2, backward = point2 to point1
        forward_azimuths, backward_azimuths, distances = geod_wgs84.inv([long1], [lat1], [long2], [lat2])

        forward_azimuth = forward_azimuths[0]
        backward_azimuth = backward_azimuths[0]
        distance = distances[0]
        return forward_azimuth, backward_azimuth, distance



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


from array import array
import datetime


class GeoPoint:
    x = 0
    y = 0

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def toString(self):
        return str(self.x) + ' ' + str(self.y)

class GeoRequest:
    points = []
    productStartDate = None
    productEndDate = None

    def setStartDate(self, date: datetime.datetime):
        self.productStartDate = date
        return self

    def setEndDate(self, date: datetime.datetime):
        self.productStartDate = date
        return self    


    def hasStartDate(self):
        return self.productStartDate != None   

    def hasEndDate(self):
        return self.productEndDate != None

    def getStartDate(self):
        return self.productStartDate

    def getEndDate(self):
        return self.productEndDate              

    def fromBoundingBox(self,boundingBox):
        minLatLon = [boundingBox[0], boundingBox[1]]
        maxLatLng = [boundingBox[2], boundingBox[3]]
        self.append(GeoPoint(minLatLon[1], minLatLon[0]))
        self.append(GeoPoint(maxLatLng[1], minLatLon[0]))
        self.append(GeoPoint(maxLatLng[1], maxLatLng[0]))
        self.append(GeoPoint(minLatLon[1], maxLatLng[0]))
        self.append(GeoPoint(minLatLon[1], minLatLon[0]))
        return self

    def fromArray(self, arrayData = []):
        for point in arrayData:
            self.append(GeoPoint(point[0], point[1]))
        return self    

    def append(self, point: GeoPoint):
        self.points.append(point)
        return self

    def toTestString(self):
        outputString = ''
        idx = 0
        for point in self.points:
            if idx > 0:
                outputString = outputString + "\n"
            outputString = outputString + str(point.y)+','+str(point.x)+',red,marker'
            idx = idx + 1
        return outputString     

        
    def toString(self):
        outputString = 'POLYGON(('
        idx = 0
        for point in self.points:
            if idx > 0:
                outputString = outputString + ','
            outputString = outputString + point.toString()
            idx = idx + 1
        outputString = outputString + '))'
        return outputString        


import math
# Semi-axes of WGS-84 geoidal reference
WGS84_a = 6378137.0  # Major semiaxis [m]
WGS84_b = 6356752.3  # Minor semiaxis [m]
class Airport: 
    id = None
    icao_code = None           
    iata_code = None
    name = None
    city = None
    country = None
    lat_deg  = None
    lat_min = None
    lat_sec = None
    lat_dir = None
    lon_deg = None
    lon_min = None
    lon_sec = None
    lon_dir = None
    altitude = None
    lat_decimal = None
    lon_decimal = None

    def  __init__(self, db_data):
        self.id = db_data[0]
        self.icao_code = db_data[1]
        self.iata_code = db_data[2]
        self.name = db_data[3]
        self.city = db_data[4]
        self.country = db_data[5]
        self.lat_deg = db_data[6]
        self.lat_min = db_data[7]
        self.lat_sec = db_data[8]
        self.lat_dir = db_data[9]
        self.lon_deg = db_data[10]
        self.lon_min = db_data[11]
        self.lon_sec = db_data[12]
        self.lon_dir = db_data[13]
        self.altitude = db_data[14]
        self.lat_decimal = db_data[15]
        self.lon_decimal = db_data[16]

    def deg2rad(self,degrees):
        return math.pi*degrees/180.0
    def rad2deg(self,radians):
        return 180.0*radians/math.pi



    # Earth radius at a given latitude, according to the WGS-84 ellipsoid [m]
    def WGS84EarthRadius(self,lat):
        # http://en.wikipedia.org/wiki/Earth_radius
        An = WGS84_a*WGS84_a * math.cos(lat)
        Bn = WGS84_b*WGS84_b * math.sin(lat)
        Ad = WGS84_a * math.cos(lat)
        Bd = WGS84_b * math.sin(lat)
        return math.sqrt( (An*An + Bn*Bn)/(Ad*Ad + Bd*Bd) )

    def boundingBox(self,halfSideInKm = 10):
        lat = self.deg2rad(self.lat_decimal)
        lon = self.deg2rad(self.lon_decimal)
        halfSide = 1000*halfSideInKm

        # Radius of Earth at given latitude
        radius = self.WGS84EarthRadius(lat)
        # Radius of the parallel at given latitude
        pradius = radius*math.cos(lat)

        latMin = lat - halfSide/radius
        latMax = lat + halfSide/radius
        lonMin = lon - halfSide/pradius
        lonMax = lon + halfSide/pradius

        return [self.rad2deg(latMin), self.rad2deg(lonMin), self.rad2deg(latMax), self.rad2deg(lonMax)] 


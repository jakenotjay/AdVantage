import os.path
import sqlite3
from .models.Airport import Airport

AIRPORT_DB = 'global_airports_sqlite.db'

class Airports:
    connection = None
    
    def __init__(self):
        self.connect()

    def connect(self):
        if self.connection == None:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(BASE_DIR, AIRPORT_DB)
            self.connection = sqlite3.connect(db_path)

    def close(self):
        if self.connection != None:
            self.connection.close()
            self.connection = None 
        return self           

    def findByICAOCode(self, code):
        self.connect()
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM airports WHERE icao_code = ? LIMIT 1", [code])
        results = cur.fetchall()
        if len(results) == 1:
            return Airport(results[0])
        else:
            return None  

    def findByIATACode(self, code):
        self.connect()
        cur = self.connection.cursor()
        cur.execute("SELECT * FROM airports WHERE iata_code = ? LIMIT 1", [code])
        results = cur.fetchall()
        if len(results) == 1:
            return Airport(results[0])
        else:
            return None            

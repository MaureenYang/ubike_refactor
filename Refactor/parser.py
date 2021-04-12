import os,sys
import pymongo
from pymongo import MongoClient
import pandas as pd

sys.path.append("parser/")
from youbike_processor import YoubikeProcessor
from weather_processor import WeatherProcessor

debug = False

if debug:
    def printfunc(*x):
        print(x)
else:
    def printfunc(*x):
        pass

class data_parser():

    bike_parser = None
    weather_parser = None
    weather_parser2 = None
    bdata = None
    wdata = None
    wdata2 = None

    def cal_dist(self, coor1, coor2):
        return (coor1['x']-coor2['x'])**2+(coor1['y']-coor2['y'])**2

    def __init__(self):
        self.bike_parser = YoubikeProcessor()
        self.weather_parser = WeatherProcessor()
        self.weather_parser2 = WeatherProcessor()

    def update_current_data(self):
        self.bike_parser.read('raw_data/YouBikeTPNow.json')
        self.bdata = self.bike_parser.get_dict()
        printfunc('----------data 1----------')
        printfunc(self.bdata)
        self.weather_parser.read('raw_data/WeatherDataNow.json')
        self.wdata = self.weather_parser.get_dict()
        printfunc('----------data 1----------')
        printfunc(self.wdata)
        self.weather_parser2.read('raw_data/WeatherDataNow2.json')
        self.wdata2 = self.weather_parser2.get_dict()
        printfunc('----------data 2----------')
        printfunc(self.wdata2)


    def generate_bike_weather_data(self):
        newbikedata = []
        for station in self.bdata:
            min_dist = 9999
            min_station = None
            station_coor = {}
            wstation_coor = {}

            station_coor['x'] = station['lat']
            station_coor['y'] = station['lng']

            for wstation in self.wdata:
                wstation_coor['x'] = wstation['lat']
                wstation_coor['y'] = wstation['lon']
                dist = self.cal_dist(station_coor,wstation_coor)
                if dist < min_dist:
                    min_station = wstation['stationId']
                    min_dist = dist
            printfunc('min station:',min_station, ', dist:',min_dist)
            #wlist = ['PRES','TEMP','HUMD','WDSD','WDIR','H_FX','H_XD','H_24R']
            #w2list = ['D_TS','VIS','H_UVI','Weather']
            wlist = ['PRES','TEMP','HUMD','WDSE','WDIR','WSGust','WDGust','H_24R']
            w2list = ['PrecpHour','Visb','UVI','Weather']
            for wstation in self.wdata:
                if wstation['stationId'] == min_station:
                    for key, value in wstation.items():
                        if key in wlist:
                            station.update({key:value})


            min_dist2 = 9999
            min_station2 = None
            for wstation in self.wdata2:
                wstation_coor['x'] = wstation['lat']
                wstation_coor['y'] = wstation['lon']
                dist = self.cal_dist(station_coor,wstation_coor)
                if dist < min_dist2:
                    min_station2 = wstation['stationId']
                    min_dist2 = dist

            printfunc('min station2:',min_station2, ', dist:',min_dist2)
            for wstation in self.wdata2:
                if wstation['stationId'] == min_station2:
                    for key, value in wstation.items():
                        if key in w2list:
                            station.update({key:value})

            newbikedata = newbikedata  + [station]

        # to csv
        # to database
        df = pd.DataFrame(newbikedata)
        df.to_csv('bike_weather.csv')
        return df

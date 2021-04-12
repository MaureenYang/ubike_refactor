import os,sys
import pymongo
import pandas as pd
import datetime

'''
class parser
1. parse information from raw data and insert to database
2. get information from database
'''

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

    '''database'''
    dbclient = None
    ubikedb = None
    ubikecol = None
    weathercol = None
    weather2col = None
    parsedcol = None

    def __init__(self):
        self.bike_parser = YoubikeProcessor()
        self.weather_parser = WeatherProcessor()
        self.weather_parser2 = WeatherProcessor()

    def cal_dist(self, coor1, coor2):
        return (coor1['x']-coor2['x'])**2+(coor1['y']-coor2['y'])**2

    def parsing_bike_data(self,path,insert=True):
        self.bike_parser.read(path)
        self.bdata = self.bike_parser.get_dict()
        if insert:
            for x in self.bdata:
                idx = self.ubikecol.insert_one(x)

        return self.bdata

    def parsing_weather_data(self,path,insert=True):
        self.weather_parser.read(path)
        self.wdata = self.weather_parser.get_dict()
        if insert:
            for x in self.wdata:
                idx = self.weathercol.insert_one(x)
        return self.wdata

    def parsing_weather2_data(self,path,insert=True):
        self.weather_parser2.read(path)
        self.wdata2 = self.weather_parser2.get_dict()
        if insert:
            for x in self.wdata2:
                idx = self.weather2col.insert_one(x)
        return self.wdata2

    def insert_current_data(self):
        self.bike_parser.read('raw_data/YouBikeTPNow.json')
        self.bdata = self.bike_parser.get_dict()
        for x in self.bdata:
            idx = self.ubikecol.insert_one(x)

        self.weather_parser.read('raw_data/WeatherDataNow.json')
        self.wdata = self.weather_parser.get_dict()
        for x in self.wdata:
            idx = self.weathercol.insert_one(x)

        self.weather_parser2.read('raw_data/WeatherDataNow2.json')
        self.wdata2 = self.weather_parser2.get_dict()
        for x in self.wdata2:
            idx = self.weather2col.insert_one(x)

    def generate_bike_weather_data(self,dbinsert=False,csvgen=False):
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
            if csvgen:
                newbikedata = newbikedata  + [station]

            if dbinsert:
                x = self.parsedcol.insert_one(station)
                print(x)

        if csvgen:
            df = pd.DataFrame(newbikedata)
            df.to_csv('bike_weather.csv')

    def connect2db(self):
        self.dbclient = pymongo.MongoClient("mongodb://localhost:27017/")
        self.ubikedb = self.dbclient["YoubikeDB"]

        self.ubikecol = self.ubikedb["new_ubike_col"]
        self.weathercol = self.ubikedb["new_weather_col"]
        self.weather2col = self.ubikedb["new_weather2_col"]
        self.parsedcol = self.ubikedb["parsed_col"]

        return self.ubikedb

    def dbclose(self):
        self.dbclient.close()

    def db_iterate_find(self,col,query,batchsize=2000,order='asc',hint=None):

        _query = query
        _re_df = pd.DataFrame()
        _sort =  ('_id', -1)
        if order == "asc":
            _sort =  ('_id', 1)

        while True:
            if hint:
                queryResults = col.find(_query).sort([_sort]).limit(batchsize).hint(hint)
            else:
                queryResults = col.find(_query).sort([_sort]).limit(batchsize)

            df = pd.DataFrame(list(queryResults))
            _re_df = _re_df.append(df)
            query_cnt = len(df)

            print('query_cnt:',query_cnt)
            if query_cnt == 0:
                break

            if query_cnt < batchsize:
                break

            last_id = df['_id'].iloc[-1]
            _query['_id'] = {
                '$lt': last_id
            }

            if order == "asc":
                _query['_id'] = {
                    '$gt': last_id
                }

        _re_df.to_csv("find_result.csv")



if __name__ == "__main__":

    p = data_parser()

    ubikedb = p.connect2db()
    #ubikecol = ubikedb["new_ubike_col"]
    #ubikecol = ubikedb["ubike_data_min"]

    srcfilepath = "raw_data/youbike_data/"
    filesinpath = os.listdir(srcfilepath)

    '''
    for f in sorted(filesinpath):
        try:
            print(f)
            p.parsing_bike_data(srcfilepath + f, True)
        except Exception as e:
            print(e)

    srcfilepath = "raw_data/weather_data/"
    filesinpath = os.listdir(srcfilepath)

    for f in sorted(filesinpath):
        try:
            print(f)
            p.parsing_weather_data(srcfilepath + f, True)
        except Exception as e:
            print(e)

    srcfilepath = "raw_data/weather2_data/"
    filesinpath = os.listdir(srcfilepath)
    for f in sorted(filesinpath):
        try:
            print(f)
            p.parsing_weather2_data(srcfilepath + f, True)
        except Exception as e:
            print(e)
    '''
    # get data for all stations in last 12 hours
    # input start date, how many hours, station no.

    '''
    #d = ubikecol.find({'sno':1})
    start = datetime.datetime(2018, 7, 31, 00, 00, 00)
    tdelta = datetime.timedelta(hours=12)
    #end = datetime.datetime(2021, 4, 1, 00, 00, 00)
    end = start + tdelta
    #d = ubikecol.find( {'sno': 1, 'mday': {'$lt': end, '$gte': start}}).limit(3)
    #query = {'sno': 1, 'mday': {'$lt': end, '$gte': start},}
    query = {'mday': {'$lt': end, '$gte': start}}
    #query = {'sno': 1}
    p.db_iterate_find(ubikecol,query)
    '''

    #for x in d:
    #    print(x)

    p.dbclose()

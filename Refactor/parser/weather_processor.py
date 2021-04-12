#
#   filename: weather_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#

import os
import gzip
import json
import xml.etree.ElementTree
import datetime,time
import dateutil.parser
from data_processor import data_processor


def getkeyvalue(src): #get dict and return dict
    new_element = {}
    key_name = src['elementName']
    try:
        new_element[key_name] = float(src['elementValue']['value'])
    except:
        #print('not number:',src['elementValue']['value'])
        new_element[key_name] = src['elementValue']['value']
    return new_element

class WeatherProcessor(data_processor):
    """
        this is description of Weather processor
    """
    __type__ = 'WeatherProcessor'
    #stationIdList = ['C0AH70', 'C0AC80', #'C0A9F0','C0A9C0','C0A9E0','C0A980','C0AC40','C0AH40','C0AI40','466930','466910','466920','CAAH60','CAA090','CAA040']

    stationIdList = ['C0AH70', 'C0AC80', 'C0A9F0','C0A9C0','C0A9E0','C0A980','C0AC40','C0AH40','C0AI40','466910','466920']

    weatherPara = ['PRES','TEMP','HUMD','WDSD','WDIR','H_FX','H_XD','H_24R','24R','D_TS','VIS','H_UVI']
    weatherPara_map = {'PRES':'PRES','TEMP':'TEMP','HUMD':'HUMD','WDSD':'WDSE','WDIR':'WDIR','H_FX':'WSGust','H_XD':'WDGust','H_24R':'H_24R','24R':'24R','D_TS':'PrecpHour','VIS':'Visb','H_UVI':'UVI'}

    def __init__(self):
        super().__init__()

    def read(self,filename):
        try:
            json_file = open(filename)
            dict_data = json.load(json_file)
            json_file.close()
            dict_tmp = []

            for key,value in dict_data['cwbopendata'].items():
                try:
                    if(key == 'sent'):
                        time = datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S+08:00")
                    if(key =='location'):
                        taipei_list = self.find_taipei(value)
                        dict_tmp = self.station_parsing(taipei_list)

                except Exception as e:
                    print(e)
                    print('error: ',value)
                    dict_tmp = None

            self.__data_dict__ = dict_tmp

        except os.error as e:
            self.__data_dict__ = None


    def get_dict(self):
        return self.__data_dict__

    def station_parsing(self,list):
        stationlist = []
        for data in list:
            newdata = {}
            newdata['stationId'] = data['stationId']
            newdata['locationName'] = data['locationName']
            newdata['lat'] = float(data['lat'])
            newdata['lon'] = float(data['lon'])
            newdata['time'] = datetime.datetime.strptime(data['time']['obsTime'], "%Y-%m-%dT%H:%M:%S+08:00")
            wdata = self.weather_element(data['weatherElement'])
            newdata.update(wdata)
            stationlist = stationlist + [newdata]

        return stationlist

    def weather_element(self,list):
        ele_dict = {}
        for data in list:
            if data['elementName'] in self.weatherPara:
                ele_dict[self.weatherPara_map[data['elementName']]] = getkeyvalue(data)[data['elementName']]

        return ele_dict

    def find_taipei(self,list):
        tmplist = []
        for data in list:
            if data['stationId'] in self.stationIdList:
                tmplist = tmplist + [data]

        return tmplist

    def find_taipei_id(self,list):
        for data in list:
            foundflag = False
            pdata = data['parameter']
            for ppdata in pdata:
                if ppdata['parameterName'] == "CITY" and ppdata['parameterValue'] == "臺北市":
                    foundflag = True
            if foundflag:
                print(data['stationId'])

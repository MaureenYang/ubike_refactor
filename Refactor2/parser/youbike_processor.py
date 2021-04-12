#
#   filename: youbike_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#   todo: you can parse the item you want only to the dict
#

import os,sys
import gzip
import json
import datetime,time
import dateutil.parser
from data_processor import data_processor

class YoubikeProcessor(data_processor):
    """
        this is description of Youbike processor
    """
    __type__ = 'YoubikeProcessor'

    def __init__(self):
        super().__init__()

    def read(self,filename):
        try:
            json_file = open(filename)
            dict_data = json.load(json_file)
            json_file.close()

            dict_tmp = []
            for key,value in dict_data['retVal'].items():
                if int(value['sno']) == 988:
                    continue;
                try:
                    value['sno'] = int(value['sno'])
                    value['tot'] = int(value['tot'])
                    value['sbi'] = int(value['sbi'])
                    value['bemp'] = int(value['bemp'])
                    value['lat'] = float(value['lat'])
                    value['lng'] = float(value['lng'])
                    value['act'] = int(value['act'])
                    value['mday'] = datetime.datetime.strptime(value['mday'], "%Y%m%d%H%M%S")
                    dict_tmp = dict_tmp +[value]
                except Exception as e:
                    print(e)
                    print('error: ',value)
            self.__data_dict__ = dict_tmp

        except os.error as e:
            self.__data_dict__ = None



    def get_dict(self):
        return self.__data_dict__

#
#   filename: data_processor.py
#       A class to read data from file, parse data to dict structure and save data to MongoDB
#   auther: Maureen Yang
#   data: 2018/07/18
#

class data_processor():
    """
        this is description of data processor
    """
    __type__ = 'Data Processor'
    __data_dict__ = None

    def __init__(self):
        pass

    def read(self):
        print(self.__type__,' read function')

    def get_dict(self):
        print(self.__type__,' get dict function')

    def save2DB(self):
        print(self.__type__,' save to DB function')

import urllib.request
import time
import threading
import datetime
import os, sys
import shutil
import data_center


ONE_MIN = 60
ONE_HOUR = ONE_MIN * 60
ONE_DAY = ONE_HOUR * 24


def GetUbikeDataThread(p):

    ubike_url = "https://tcgbusfs.blob.core.windows.net/blobyoubike/YouBikeTP.json"
    while True:
        try:
            urllib.request.urlretrieve(ubike_url, "YouBikeTP.json")

            try:
               shutil.copyfile("YouBikeTP.json","raw_data/YouBikeTPNow.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except Exception as e:
               print("Unexpected error:", sys.exc_info())
               print("error:",e)

            '''
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "YouBikeTP_" + st + ".json"
            os.rename("YouBikeTP.json",fname)
            shutil.move(fname,"raw_data/youbike_data/")
            print("get file : "+fname)
            #p.parsing_bike_data("raw_data/YouBikeTPNow.json")
            '''
            time.sleep(ONE_MIN)    #every minites
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Youbike HTTPerror]")
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Youbike URLerror]")
            time.sleep(10)
        except TimeoutError as e:
            print("[Youbike Timeouterror]")
            time.sleep(10)
        except:
            print("[Youbike]Unexpected Error!")
            time.sleep(10)


def GetWeatherThread(p):
    uvdata = -1
    weather_url = "http://opendata.cwb.gov.tw/opendataapi?dataid=O-A0001-001&authorizationkey=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725&downloadType=WEB&format=JSON"
    while True:
        try:
            res = urllib.request.urlretrieve(weather_url,"weather_data.json")
            try:
               shutil.copyfile("weather_data.json","raw_data/WeatherDataNow.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except:
               print("Unexpected error:", sys.exc_info())
            '''
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "weather_data_" + st + ".json"
            os.rename("weather_data.json",fname)
            shutil.move(fname,"raw_data/weather_data/")
            print("get file : "+fname)
            p.parsing_weather_data("raw_data/WeatherDataNow.json")
            '''
            time.sleep(ONE_HOUR) #every hour

        except TimeoutError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather error]time out! try again")
            time.sleep(30)
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather HTTP error]")
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Weather URL error]")
            time.sleep(10)

def GetWeather2Thread(p):
    uvdata = -1
    #weather_url = "http://opendata.cwb.gov.tw/opendataapi?dataid=O-A0001-001&authorizationkey=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725"
    weather_url = "https://opendata.cwb.gov.tw/fileapi/v1/opendataapi/O-A0003-001?Authorization=CWB-B74D517D-9F7C-44B9-90E9-4DF76361C725&downloadType=WEB&format=JSON"
    while True:
        try:
            res = urllib.request.urlretrieve(weather_url, "weather_data2.json")
            try:
               shutil.copyfile("weather_data2.json","raw_data/WeatherDataNow2.json")
            except IOError as e:
               print("Unable to copy file. %s" % e)
            except:
               print("Unexpected error:", sys.exc_info())
            '''
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "weather2_data_" + st + ".json"
            os.rename("weather_data2.json",fname)
            shutil.move(fname,"raw_data/weather2_data/")
            print("get file : "+fname)
            p.parsing_weather2_data("raw_data/WeatherDataNow2.json")
            '''
            time.sleep(ONE_HOUR) #every hour

        except TimeoutError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather error]time out! try again")
            time.sleep(30)
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Weather HTTP error]")
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Weather URL error]")
            time.sleep(10)

def GetAQIThread():

    aqi_url = "http://opendata2.epa.gov.tw/AQI.json"
    while True:
        try:
            urllib.request.urlretrieve(aqi_url, "aqi.json")
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "aqi_" + st + ".json"
            os.rename("aqi.json",fname)
            shutil.move(fname,"aqi_data")

            print("get file : "+fname)
            time.sleep(ONE_HOUR)
        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[AQI HTTPerror]")
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[AQI URLerror]")
            time.sleep(10)
        except TimeoutError as e:
            print("[AQI Timeouterror]")
            time.sleep(10)
        except:
            print("[AQI]Unexpected Error!")
            time.sleep(10)
def GetAirboxThread():

    airbox_url = "https://tpairbox.blob.core.windows.net/blobfs/AirBoxData_V3.gz"
    while True:
        try:
            urllib.request.urlretrieve(airbox_url, "airbox_data.gz")
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y_%m_%d_%H_%M_%S')
            fname = "airbox_data_" + st + ".gz"
            os.rename("airbox_data.gz",fname)
            shutil.move(fname,"airbox_data")
            print("get file : "+fname)
            time.sleep(ONE_HOUR) #every hour

        except urllib.error.HTTPError as e:
            # Maybe set up for a retry, or continue in a retry loop
            print("[Airbox HTTPerror]")
            time.sleep(10)
        except urllib.error.URLError as e:
            # catastrophic error. bail.
            print("[Airbox URLerror]")
            time.sleep(10)
        except TimeoutError as e:
            print("[Airbox Timeouterror]")
            time.sleep(10)
        except:
            print("[Airbox]Unexpected Error!")
            time.sleep(10)


#### main.py
if __name__ == "__main__":

    print ("Crawler Starting...")
    p = data_center.data_parser()
    p.connect2db()

    #create thread
    ubike_thread = threading.Thread(target = GetUbikeDataThread, args = (p,))
    weather_thread = threading.Thread(target = GetWeatherThread, args = (p,))
    weather2_thread = threading.Thread(target = GetWeather2Thread, args = (p,))

    ubike_thread.start()
    weather_thread.start()
    weather2_thread.start()

    ubike_thread.join()
    weather_thread.join()
    weather2_thread.join()

    p.dbclose()
    print ("Crawler Finished")

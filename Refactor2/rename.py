import os,sys
import re
srcfilepath = "E:/csvfile/source/"
filesinpath = os.listdir(srcfilepath)



for f in sorted(filesinpath):
    try:
        print(f)
        namelist = f.split(".")

        num_sec = namelist[0]
        numb = str(int(re.search(r'\d+', num_sec).group())).zfill(3)
        newfilename = "data_sno_"+numb+".csv"
        print(newfilename)
        os.rename(srcfilepath + f,srcfilepath + newfilename)
    except Exception as e:
        print(e)

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 21:50:32 2019

@author: 15899

将从IRIS上下载的tar解压得到的dataless文件提取出台站信息:
    stationName netName longitude latitude elevation
"""

import glob,os
from obspy.io.xseed import Parser

def ProcessDataless(Dataless,fp):
    blk = Dataless.blockettes
    stationNumber = len(blk[50])
    for ii in range(stationNumber):
        stationName = blk[50][ii].station_call_letters
        netName = blk[50][ii].network_code
        latitude = blk[50][ii].latitude
        longitude = blk[50][ii].longitude
        elevation = blk[50][ii].elevation
        information = [stationName,netName,str(latitude),str(longitude),str(elevation)]
        print(information)
        WriteInformation(information,fp)
        
def WriteInformation(information,fp):
    [stationName,netName,latitude,longitude,elevation] = information
    fp.write(stationName+ ' ' +netName+ ' ' +latitude+ ' ' +longitude + ' ' + elevation + '\n')
  
DATALESS_PATH = '/shdisk/lab2/zgh/IBBN_TIRR/mseed/'
SAVE_PATH = '/shdisk/lab2/zgh/IBBN_TIRR/stationInformation.txt'
DATALESS = sorted(glob.glob(DATALESS_PATH+'*.dataless'))

if not os.path.exists(os.path.split(SAVE_PATH)[0]):
    os.makedirs(os.path.split(SAVE_PATH)[0])
fp = open(SAVE_PATH,'w')
for dataless in DATALESS:
   print(dataless.split('/')[-1]) 
   Dataless = Parser(dataless)
   ProcessDataless(Dataless,fp)
fp.close()
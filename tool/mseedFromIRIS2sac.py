# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 21:50:32 2019

@author: 15899

将从IRIS上下载的数据转为按天分割的数据
"""

import os,io,glob,datetime
import numpy as np
from obspy import read
from obspy.core import UTCDateTime
from obspy.core.stream import Stream
import obspy
from concurrent.futures import ThreadPoolExecutor

def SplitMseed(mseed):
    sta = mseed.split('/')[-1]
    sta,net = sta.split('.')[0:2]
    tmp_workDir = mseedPath + sta + '-' + net + '/'
    if not os.path.exists(tmp_workDir):
        os.makedirs(tmp_workDir)
    os.chdir(tmp_workDir)
    if len(glob.glob(tmp_workDir+'*' + channel + '.SAC'))==0:
        os.system('mseed2sac -dr '+mseed)
    sacfile = sorted(glob.glob(tmp_workDir+'*' + channel + '.SAC'))
    print('this mseed file convert to '+str(len(sacfile))+' sac file')
    for sac in sacfile:
        st = read(sac)
        ProcessStream(st)
        os.remove(sac)
    os.chdir(mseedPath)
    os.removedirs(tmp_workDir)

def ProcessStream(stream):
    n_day = 0
    for trace in stream:
        if trace.stats.channel != channel:
            continue
        net = trace.stats.network
        sta = trace.stats.station
        station_path = sacPath + sta + '-' + net
        if not os.path.exists(station_path):
            os.makedirs(station_path)    
        starttime = trace.stats.starttime
        endtime = trace.stats.endtime
        day_gap = datetime.date(endtime.year,endtime.month,endtime.day) - datetime.date(starttime.year,starttime.month,starttime.day)
        time_day = [starttime]
        for ii in range(day_gap.days):
            splittime = datetime.date(time_day[ii].year,time_day[ii].month,time_day[ii].day) + datetime.timedelta(1)
            splittime = str(splittime.year)+'-'+str(splittime.month)+'-'+str(splittime.day)+'T00:00:00'
            splittime = UTCDateTime(splittime)
            time_day.append(splittime)
        time_day.append(endtime)
        
        for ii in range(len(time_day)-1):
            starttime = time_day[ii]
            endtime = time_day[ii+1]
            st_day = trace.slice(starttime,endtime,nearest_sample=False)
            if st_day.stats.npts < 10:
                continue
            day_name = sacPath + sta + '-' + net + '/' + sta + '-' + net + '-' + str(starttime.year) + '-' + str(starttime.julday).zfill(3) + '-' + channel + '.SAC'
            if os.path.exists(day_name):
                st_tmp = read(day_name,format='SAC')
                if int(st_tmp[0].stats.sampling_rate)!=int(resamplingRate):
                    resampling_rate = int(round(st_tmp[0].stats.sampling_rate/resamplingRate))
                    st_tmp[0].decimate(resampling_rate)
                day_name_tmp = day_name+'.tmp'
                st_day.write(day_name_tmp,format='SAC')
                st_day = read(day_name_tmp,format='SAC')
                os.remove(day_name_tmp)
                if int(st_day[0].stats.sampling_rate)!=int(resamplingRate):
                    resampling_rate = int(round(st_day[0].stats.sampling_rate/resamplingRate))
                    if resampling_rate == 100:
                        st_day[0].decimate(10,no_filter=True)
                        st_day[0].decimate(10,no_filter=True)
                    else:
                        st_day[0].decimate(resampling_rate,no_filter=True)
                if np.abs(st_day[0].stats.sampling_rate - resamplingRate) > 0.01:
                    st_day[0].resample(resamplingRate)
                else:
                    st_day[0].stats.sampling_rate = resamplingRate
                st_tmp[0].data = st_tmp[0].data.astype(np.float32)
                st_day[0].data = st_day[0].data.astype(np.float32)                
                merge_data = Stream()
                merge_data.append(st_tmp[0])
                merge_data.append(st_day[0])
                merge_data.sort(['starttime'])
                merge_data.merge(method=1,fill_value = 'latest')
                os.remove(day_name)
                merge_data.write(day_name,format='SAC')
            else:
                if int(st_day.stats.sampling_rate)!=int(resamplingRate):
                    resampling_rate = int(round(st_day.stats.sampling_rate/resamplingRate))
                    if resampling_rate==100:
                        st_day.decimate(10,no_filter=True)
                        st_day.decimate(10,no_filter=True)
                    else:
                        st_day.decimate(resampling_rate,no_filter=True)  
                if np.abs(st_day.stats.sampling_rate - resamplingRate) > 0.01:
                    st_day.resample(resamplingRate)
                else:
                    st_day.stats.sampling_rate = resamplingRate      
                st_day.write(day_name,format='SAC')
                n_day = n_day + 1

            print(sta + ' now julday is: ' + str(starttime.date),n_day)    


def mseedFileMain(mseed):
   if os.path.getsize(mseed)<2**31:
        st = read(mseed)
        ProcessStream(st)
   else:
       print('this mseed file is larger than 2GB, so will split this file with mseed2sac routine')
       SplitMseed(mseed)

sacPath = '/shdisk/lab2/zgh/IBBN_TIRR/sac_1Hz/'
mseedPath = '/shdisk/lab2/zgh/IBBN_TIRR/mseed/'
mseedFile = glob.glob(mseedPath + '*.mseed')
resamplingRate = 1.0
channel = 'BHZ'


for mseed in mseedFile:
    mseedFileMain(mseed)
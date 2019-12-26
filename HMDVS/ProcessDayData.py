#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:48:25 2019

@author: zgh
"""

import glob,os
import numpy as np
import configparser
from obspy import read ,read_inventory
import multiprocessing as mp
from obspy.signal import filter
from obspy.signal import util
import matplotlib.pyplot as plt

def ReadPazFile(pazfile):
    with open(pazfile,'r') as fp:
        line = fp.readline().strip()
        gain = float(line)
        line = fp.readline().strip()
        Numzeros = int(line)
        zeros = []
        for ii in range(Numzeros):
            line = fp.readline().strip()
            line = line.split('\t')
            zeros.append(float(line[0]) + float(line[1])*1j)
            
        line = fp.readline().strip()   
        Numpoles = int(line)
        poles = []
        for ii in range(Numpoles):
            line = fp.readline().strip()
            line = line.split('\t')
            poles.append(float(line[0]) + float(line[1])*1j)
            
            
    paz_sts2 = {
    'poles': poles,
    'zeros': zeros,
    'gain': gain,
    'sensitivity': 1261.8}
    
    return paz_sts2

def RemoveInstruction(trace,resp_file):
    if resp_file == []:
        print('no resp file for remove instrument response')
        exit()
    else:
        if resp_file.split('/')[-1].split('.')[0] == "RESP":
            pre_filt = (0.0005, 0.001, 0.45, 0.5)
            inv = read_inventory(resp_file)
            trace.remove_resp(inventory=inv,pre_filt=pre_filt, output="VEL",water_level=60, plot=True)
            
        if resp_file.split('/')[-1].split('.')[0] == "PZ":
            paz_sts2 = ReadPazFile(resp_file)
#            paz_1hz = corn_freq_2_paz(1.0, damp=0.707)  # 1Hz instrument
#            paz_1hz['sensitivity'] = 1.0
            trace.simulate(paz_remove=None, paz_simulate=paz_sts2)
            
        return trace

def SpecWhiten(trace):    
    if specWhiten_method == 0:
        return trace
    else:
        dataFFT = np.fft.fft(trace.data)
        if specWhiten_method == 1:
            envelope = filter.envelope(np.abs(dataFFT))
            dataFFT = dataFFT/envelope
        else:
            dataFFT = dataFFT/np.abs(dataFFT)
        
        trace.data = np.real(np.fft.ifft(dataFFT))
        
    return trace
        
def TemporalNormalization(trace):
    if normalization_method == 1:
        trace.data = np.sign(trace.data)
        
    if normalization_method == 2:
        winsize = period_max / 2.0 * trace.stats.sampling_rate
        tmp = util.smooth(abs(trace.data),round(winsize))
        trace.data = trace.data/tmp
    
    return trace

def Preprocess(sta_net):
    if not os.path.exists(sac_path + sta_net):
        print('this station does not exists!!!')
        return
    
    data_file = sorted(os.listdir(sac_path + sta_net))
    if os.path.exists(pre_save_path + sta_net) and len(data_file)==len(os.listdir(pre_save_path + sta_net)):
        return
    
    if not data_file == [] and not os.path.exists(pre_save_path + sta_net):
        os.makedirs(pre_save_path + sta_net)
    
    flag_removeResp = True
    if flag_resp == 1 or flag_resp == 2:
        resp_file = glob.glob(resp_path + '*' + sta_net.split('-')[1] + '-' +  sta_net.split('-')[0] + '*.' + channel)
        if resp_file == [] and flag_resp == 2:
           flag_removeResp = False
    else:
        flag_removeResp = False
    
    for file in data_file:
        save_file = pre_save_path + sta_net + '/' + file.split('.')[0] + '.SAC'
        if os.path.exists(save_file):
            continue
        
        stream = read(sac_path + sta_net + '/' + file)
        if len(stream) != 1:
            print('file:' + file + ' have more than one trace in the stream.')
            continue
        
        trace = stream[0]
#        trace.plot(title = trace.id + 'origin data')
        if trace.stats.npts < 3600.0 * trace.stats.sampling_rate:
            continue

        trace = trace.detrend(type='linear')   # de mean
        trace = trace.detrend(type = 'constant')   # de trend        
#        trace.plot(title = trace.id + ' after demean and detrend')

        # for remove resp
        if flag_removeResp:
            RemoveInstruction(trace,resp_file[0])
#            trace.plot(title = trace.id + ' after remove instrument response')

        
        if np.abs(trace.stats.sampling_rate - samplate_rate) > 0.01:
            trace.resample(samplate_rate)
#            trace.plot(title = trace.id + ' after resample')
        
        
        trace.data = filter.bandpass(trace.data,1.0/period_max,1.0/period_min,trace.stats.sampling_rate,zerophase=True)
#        trace.plot(title = trace.id + ' after badpass for ' + str(1.0/period_max) + '~' + str(1.0/period_min))
        
        trace = TemporalNormalization(trace)
#        trace.plot(title = trace.id + ' after Temporal Normalization')  

        trace = SpecWhiten(trace)
#        trace.plot(title = trace.id + ' after SpecWhiten')      
        
        stream[0] = trace
        print(save_file)
        stream.write(save_file)

def GetStackDay(sta_pair):
    day_1 = [x.split('/')[-1].split('-')[2] + '-' + x.split('/')[-1].split('-')[3] for x in sorted(glob.glob(pre_save_path  + sta_pair[0] + '/*'))]
    day_2 = [x.split('/')[-1].split('-')[2] + '-' + x.split('/')[-1].split('-')[3] for x in sorted(glob.glob(pre_save_path  + sta_pair[1] + '/*'))]
    stackDay = [same_day for same_day in day_1 if same_day in day_2]
    
    return stackDay

def SplitData(file_1,file_2):
    trace_1 = read(file_1)[0]
    trace_2 = read(file_2)[0]
    sta_1_begin = int(np.floor((trace_1.stats.sac['nzhour']*3600+trace_1.stats.sac['nzmin']*60+trace_1.stats.sac['nzsec']+trace_1.stats.sac['nzmsec']*0.001)*samplate_rate))
    sta_2_begin = int(np.floor((trace_2.stats.sac['nzhour']*3600+trace_2.stats.sac['nzmin']*60+trace_2.stats.sac['nzsec']+trace_2.stats.sac['nzmsec']*0.001)*samplate_rate))
    
    fold_point = int(np.floor(0.5*split_hour*3600*samplate_rate))
    split_point = int(np.floor(split_hour*3600*samplate_rate))
    
    sta_1_end = sta_1_begin  + trace_1.stats.npts - 1
    sta_2_end = sta_2_begin  + trace_2.stats.npts - 1
    
    if sta_1_begin > sta_2_end or sta_2_begin > sta_1_end:   # do't have the overlap time
        return [[],[]]
    
    if sta_1_begin >= sta_2_begin:
        begin1 = 0
        begin2 = sta_1_begin - sta_2_begin
    else:
        begin2 = 0
        begin1 = sta_2_begin - sta_1_begin
    
    if sta_1_end >= sta_2_end:
        end2 = trace_2.stats.npts
        end1 = begin1 + end2 - begin2
    else:
        end1 = trace_1.stats.npts
        end2 = begin2 + end1 - begin1
        
    data1_tmp = trace_1.data[begin1:end1]
    data2_tmp = trace_2.data[begin2:end2]
    
    if len(data1_tmp) < split_point:
        sta_1_data = []
        sta_2_data = []
        return [sta_1_data,sta_2_data]
    
    k = 0
    flag = True
    while flag:
        begin_ind = k*(split_point - fold_point)
        end_ind = begin_ind + split_point
        if end_ind>len(data1_tmp):
            flag = False
        else:
            k = k+1
        
    sta_1_data = np.zeros([k,split_point])
    sta_2_data = np.zeros([k,split_point])
    
    for ii in range(k):
        begin_ind = ii*(split_point - fold_point)
        end_ind = begin_ind + split_point
        sta_1_data[ii,:] = data1_tmp[begin_ind:end_ind]
        sta_2_data[ii,:] = data2_tmp[begin_ind:end_ind]
    
    return [sta_1_data,sta_2_data]

def Crosscorrelation(sta_pair):
    if len(glob.glob(cross_save_path + sta_pair[0] + '-' + sta_pair[1] + '*.npy')) < 1:
        stack_day = GetStackDay(sta_pair)
        if len(stack_day) >= config.getint('crosscorrelation','min_crosscorrelation_day'):
            stack_t_cf = np.zeros([1,int(np.floor(split_hour*3600*samplate_rate))])
            ncf = 0
            ndaycf = 0
            for day in stack_day:
                file_1 = pre_save_path + sta_pair[0] + '/' + sta_pair[0] + '-' + day + '-' + channel + '.SAC'
                file_2 = pre_save_path + sta_pair[1] + '/' + sta_pair[1] + '-' + day + '-' + channel + '.SAC'
                data1_data2 = SplitData(file_1,file_2)
                if len(data1_data2[0]) == 0:
                    continue
                for ii in range(len(data1_data2[0])):
                    fft1 = np.fft.fft(data1_data2[0][ii,:])
                    fft2 = np.fft.fft(data1_data2[1][ii,:])
                    f_cf = np.conj(fft1)*fft2 + np.conj(fft2)*fft1
                    f_cf = f_cf/abs(f_cf)
#                    t_cf = np.real(np.fft.ifft(f_cf))
                    t_cf = np.real(f_cf)
                    if sum(np.isnan(t_cf)) == 0:
                        t_cf = t_cf/max(abs(t_cf))
                        stack_t_cf = stack_t_cf + t_cf
                        ncf = ncf + 1
                ndaycf = ndaycf + 1
                print(day,ndaycf)
            if ncf > 0:
                stack_t_cf = stack_t_cf/max(abs(stack_t_cf.T))
                save_cross = cross_save_path + sta_pair[0] + '-' + sta_pair[1] + '-' + str(ndaycf) + '-day'
                plt.plot(stack_t_cf.T),plt.show()
                np.save(save_cross, stack_t_cf)
                
def GetStation(config):
    used_sta_file = config.get('all','used_sta')
    StationUsed = []
    if  used_sta_file.split('.')[-1] == 'txt':
        with open(used_sta_file,'r') as fp:
            tmp = fp.readlines()
            for xx in tmp:
                StationUsed.append(xx.strip())
    else:
        StationUsed = os.listdir(sac_path)
       
    return sorted(StationUsed)

def GetStationPair(StationUsed):
    StationPair = []
    for ii in range(len(StationUsed)):
       for jj in range(ii + 1,len(StationUsed)):
           tmp = [StationUsed[ii],StationUsed[jj]]
           StationPair.append(tmp)
       
    return StationPair

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("Par_file.conf", encoding="utf-8")
    NumProcesser = config.getint('all','NumProcesser')
    channel = config.get('all','channel')
    projectDir = config.get('all','projectDir')
    projectName = config.get('all','projectName')
    sac_path = config.get('all','sac_path')
    pre_save_path = projectDir + projectName +'/PreprocessData/'
    cross_save_path = projectDir + projectName +'/CrosscorrelationData/'
    resp_path = config.get('preprocess','resp_path')
    period_min = config.getfloat('preprocess','period_min')
    period_max = config.getfloat('preprocess','period_max')
    samplate_rate = config.getfloat('preprocess','samplate_rate')
    flag_resp = config.getint('preprocess','flag_resp')
    specWhiten_method = config.getint('preprocess','specWhiten_method')
    normalization_method  = config.getint('preprocess','normalization_method')
    
    
    if not config.getboolean('preprocess','pre_done'):   
        if not os.path.exists(pre_save_path):
            os.makedirs(pre_save_path)
    
        StationUsed = GetStation(config)
        print('there are ' + str(len(StationUsed)) + ' in this area for ambient noise crosscorrelation.')
        print('begin to do preprocess for these origin data...')
        if NumProcesser > 1:
            pool = mp.Pool(NumProcesser)
            pool.map(Preprocess, StationUsed)
        else:
            for sta_net in StationUsed:
                Preprocess(sta_net)
    
    if not config.getboolean('crosscorrelation','cross_done'):
        if not os.path.exists(cross_save_path):
            os.makedirs(cross_save_path)
        StationPair = GetStationPair(StationUsed)
        print('begin to do crosscorrelation for these origin data...')
        split_hour = config.getfloat('crosscorrelation','split_hour')
            
        if NumProcesser > 1:
            pool = mp.Pool(NumProcesser)
            pool.map(Crosscorrelation, StationPair)
        else:
            for sta_pair in StationPair:
                Crosscorrelation(sta_pair)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob,os,shutil
import numpy as np
import configparser
from obspy import read ,read_inventory
import multiprocessing as mp
from obspy.signal import filter
from obspy.signal import util
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from obspy.signal.cross_correlation import correlate
from scipy import signal
from geopy.distance import vincenty
import h5py

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

def RemoveRap(trace):
    meanValue = np.mean(trace.data)
    stdValue = np.std(trace.data)
    thresholdValue_1 = meanValue + 4*stdValue
    thresholdValue_2 = meanValue - 4*stdValue
    index = trace.data>thresholdValue_1
    trace.data[index] = meanValue -1.5*stdValue + 3*stdValue*np.random.rand(len(trace.data[index]))
    index = trace.data<thresholdValue_2
    trace.data[index] = meanValue -1.5*stdValue + 3*stdValue*np.random.rand(len(trace.data[index]))
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
        winsize = period_max * 2.0 * trace.stats.sampling_rate
        tmp = util.smooth(abs(trace.data),round(winsize))
        trace.data = trace.data/tmp
    
    return trace

def Preprocess(sta_net):
    if not os.path.exists(sac_path + sta_net):
        print("this station's data does not exists!!!")
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
        trace = RemoveRap(trace)
        trace = trace.detrend(type='constant')   # de mean value
        trace = trace.detrend(type='linear')   # de trend      
#        trace.plot(title = trace.id + ' after demean and detrend')

        if flag_removeResp:
            RemoveInstruction(trace,resp_file[0])
#            trace.plot(title = trace.id + ' after remove instrument response')
        
        if np.abs(trace.stats.sampling_rate - samplate_rate) > 0.01:
            try:
                trace.decimate(round(trace.stats.sampling_rate/samplate_rate),no_filter=True)
            except:
                trace.resample(samplate_rate)
#            trace.plot(title = trace.id + ' after resample')                
        trace.data = filter.bandpass(trace.data,1.0/period_max,1.0/period_min,trace.stats.sampling_rate,corners = 2, zerophase=True)
#        trace.plot(title = trace.id + ' after badpass for ' + str(1.0/period_max) + '~' + str(1.0/period_min))
        trace = RemoveRap(trace)
#        trace.plot(title = trace.id + ' after RemoveRap')        
        trace = SpecWhiten(trace)
#        trace.plot(title = trace.id + ' after SpecWhiten')
        trace = RemoveRap(trace)
#        trace.plot(title = trace.id + ' after RemoveRap again')
        trace = TemporalNormalization(trace)
#        trace.plot(title = trace.id + ' after Temporal Normalization')      
        
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
    
    if split_hour == 24:
        return [data1_tmp,data2_tmp]
    else:           
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

def cxcorr(a,v):
    nom = np.linalg.norm(a)*np.linalg.norm(v)
    return fftpack.irfft(fftpack.rfft(a)*fftpack.rfft(v[::-1]))/nom

def SNRcalculate(t_cf,SNRwindow):
    if len(t_cf)%2 == 0:
        halfData = (t_cf[int(len(t_cf)/2) + 1:len(t_cf)] + t_cf[range(int(len(t_cf)/2) - 1,-1,-1)])/2
    signalData = halfData[SNRwindow[0]:SNRwindow[1]]
    noiseData = halfData[SNRwindow[2]:SNRwindow[3]]
    max_signal = max(signalData)
    min_noise_std = sum(noiseData^2)/len(noiseData)
    SNR = max_signal/min_noise_std
    return SNR

def StackFunc(stack_t_cf,t_cf,SNRwindow):
    if StackMethod == 0:
#        stack_t_cf = stack_t_cf + t_cf
        return True

    if StackMethod == 1:
        if SNRcalculate(t_cf,SNRwindow)> SNR_min:
#            stack_t_cf = stack_t_cf + t_cf
            return True

    if StackMethod == 2:
        if SNRcalculate(stack_t_cf + t_cf,SNRwindow)> SNRcalculate(stack_t_cf,SNRwindow):
#            stack_t_cf = stack_t_cf + t_cf
            return True

    return False

def calculateSNRwindow(sta_pair):
    distance = GetStationDistance(sta_pair[0],sta_pair[1],staPos)
    singal_min_arrival = distance/max_surfaceWave
    singal_max_arrival = distance/min_surfaceWave
    if singal_max_arrival > maxShift:
        return []
    if singal_max_arrival + 50 + SNR_window < maxShift:
        noise_min_arrival = singal_max_arrival + 50
        noise_max_arrival = singal_max_arrival + 50 + SNR_window
    else:
        noise_max_arrival = singal_min_arrival - 50
        noise_min_arrival = singal_min_arrival - 50 - SNR_window
    singal_left = np.floor(singal_min_arrival*samplate_rate)
    singal_right = np.floor(singal_max_arrival*samplate_rate)
    noise_left = np.floor(noise_min_arrival*samplate_rate)
    noise_right = np.floor(noise_max_arrival*samplate_rate)
    return [singal_left, singal_right, noise_left, noise_right]

def Crosscorrelation(sta_pair):
    if len(glob.glob(cross_save_path + sta_pair[0] + '-' + sta_pair[1] + '*.npy')) < 1:
        stack_day = GetStackDay(sta_pair)

        if len(stack_day) >= config.getint('crosscorrelation','min_crosscorrelation_day'):
            SNRwindow = calculateSNRwindow(sta_pair)
            ncf = 0
            ndaycf = 0
            for day in stack_day:
                file_1 = pre_save_path + sta_pair[0] + '/' + sta_pair[0] + '-' + day + '-' + channel + '.SAC'
                file_2 = pre_save_path + sta_pair[1] + '/' + sta_pair[1] + '-' + day + '-' + channel + '.SAC'
                data1_data2 = SplitData(file_1,file_2)
                if len(data1_data2[0]) == 0:
                    continue
                for ii in range(len(data1_data2[0])):
                    if np.size(data1_data2[0],0) == 1:
                        data1 = data1_data2[0][0]
                        data2 = data1_data2[1][0]
                    else:
                        data1 = data1_data2[0][ii,:]
                        data2 = data1_data2[1][ii,:]
                    if domain == 'freq':
# **********************************************************************#
#                        t_cf = cxcorr(data1,data2)
# **********************************************************************#                        
#                       t_cf = signal.fftconvolve(data1,data2,mode='full')
# **********************************************************************#                                                
#                        fft1 = np.fft.fft(data1)
#                        fft2 = np.fft.fft(data2)
#                        f_cf = np.conj(fft1)*fft2/np.sqrt((np.conj(fft1)*fft1)*(np.conj(fft2)*fft2)); 
#                        f_cf = np.conj(fft1)*fft2# + np.conj(fft2)*fft1
#                        f_cf = f_cf/abs(f_cf)
# **********************************************************************#
#                        fft1 = np.fft.fft(data1)
#                        fft2 = np.fft.fft(data2)
#                        f_cf = np.conj(fft1)*fft2/np.sqrt((np.conj(fft1)*fft1)*(np.conj(fft2)*fft2)); 
#                        f_cf = np.conj(fft1)*fft2# + np.conj(fft2)*fft1
#                        f_cf = f_cf/abs(f_cf)
# **********************************************************************#
                        fft1 = np.fft.fft(data1)
                        fft2 = np.fft.fft(data2)
                        f_cf = np.real(np.conj(fft1)*fft2)/np.sqrt(np.real(np.conj(fft1)*fft1)*np.real(np.conj(fft2)*fft2)); 
                        t_cf = f_cf
                        
#                        t_cf = np.real(np.fft.ifft(f_cf))
                    else:
                        t_cf = correlate(data1,data2, maxShift, demean=True, normalize=True, domain=domain)
                    
                    if sum(np.isnan(t_cf)) == 0:
                        t_cf = t_cf/max(abs(t_cf))
                        if ncf == 0:
                            stack_t_cf = t_cf
                            ncf = ncf + 1
                        else:
                            if StackFunc(stack_t_cf,t_cf,SNRwindow):
                                stack_t_cf = stack_t_cf + t_cf
                                ncf = ncf + 1
                if ncf > 0:
                    ndaycf = ndaycf + 1
                    print(day,ndaycf)
                if ndaycf > maxStackDay:
                    break
            if ndaycf > 0:
                stack_t_cf = stack_t_cf/max(abs(stack_t_cf.T))
                save_cross = cross_save_path + sta_pair[0] + '-' + sta_pair[1] + '-' + str(ndaycf) + '-day'
#                plt.plot(stack_t_cf.T),plt.show()
                np.save(save_cross, stack_t_cf)
                print(sta_pair[0] + '-' + sta_pair[1] + ' done')
                
def GetStation(config):
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

def ReadStaPosFile(StationUsed):
    sta_net = []
    lon = []
    lat = []
    ele = []
    with open(sta_pos_file,'r') as f:
        info = f.readline()
        while info:
            info = info.split()
            tmp = info[0] + '-' + info[1]
            if tmp in StationUsed:
                sta_net.append(info[0] + '-' + info[1])
                lon.append(float(info[2]))
                lat.append(float(info[3]))
                ele.append(float(info[4]))
            info = f.readline()
            
    staPos = [sta_net,lon,lat,ele]
    return staPos

def GetStationDistance(sta_1,sta_2,staPos):
    index1 = staPos[0].index(sta_1)
    index2 = staPos[0].index(sta_2)
    lon1 = staPos[1][index1]
    lat1 = staPos[2][index1]
    lon2 = staPos[1][index2]
    lat2 = staPos[2][index2]
    distance = vincenty((lat1,lon1), (lat2,lon2)).meters
    return distance        
        
def SortCrossFunction(correlate_file):
    if not config.getboolean('scan','sorted_done'):
        
        dataAll = []
        distanceAll = []
        StationUsed = GetStation(config)
        staPos = ReadStaPosFile(StationUsed)
        for file in correlate_file:
            info = file.split('-')
            sta_net_1 = info[0] + '-' + info[1]
            sta_net_2 = info[2] + '-' + info[3]
            stack_day = int(info[4])        
            if stack_day > min_crosscorrelation_day:
                distance = GetStationDistance(sta_net_1,sta_net_2,staPos)
                data = np.load(cross_save_path + file)
                dataAll.append(data)
                distanceAll.append(distance)
        Num = len(distanceAll)
        size = np.size(data)
        if size%2 == 0:
            halfsize = int(size/2)
        else:
            halfsize = int((size + 1)/2) 
        
        sortedData = np.zeros([Num,size])
        index = sorted(range(len(distanceAll)), key=lambda k: distanceAll[k])
        for ii in range(len(index)):
            sortedData[ii,:] = dataAll[index[ii]]
        distanceAll = np.array(sorted(distanceAll))
        FFtData = sortedData
        
#        FFtData = np.zeros([Num,size])
#        FFtData = fftpack.fft(sortedData)

        if size%2 == 0:
            FFtData = (FFtData[:,0:halfsize] + FFtData[:,range(size - 1, halfsize - 1, -1)])/2
        else:
            FFtData = (FFtData[:,0:halfsize] + FFtData[:,range(size - 1, halfsize - 2, -1)])/2
        
        time = np.linspace(-size/2/samplate_rate,size/2/samplate_rate,size)
        freq = np.linspace(1,halfsize,halfsize)/(size*samplate_rate)

        if not os.path.exists(projectDir + projectName + '/result'):
            os.makedirs(projectDir + projectName + '/result')

        saveName = projectDir + projectName + '/result/distance'
        np.save(saveName,distanceAll)
#        saveName = projectDir + projectName + '/result/time'
#        np.save(saveName,time)
        saveName = projectDir + projectName + '/result/freq'
        np.save(saveName,freq)
#        saveName = projectDir + projectName + '/result/time_CF'
#        np.save(saveName,sortedData)
        saveName = projectDir + projectName + '/result/freq_CF'
        np.save(saveName,FFtData)
    else:
        saveName = projectDir + projectName + '/result/distance.npy'
        distanceAll = np.load(saveName)
        saveName = projectDir + projectName + '/result/time.npy'
        time = np.load(saveName)
        saveName = projectDir + projectName + '/result/freq.npy'
        freq = np.load(saveName)
        saveName = projectDir + projectName + '/result/time_CF.npy'
        sortedData = np.load(saveName)
        saveName = projectDir + projectName + '/result/freq_CF.npy'
        FFtData = np.load(saveName)
        size = np.size(sortedData,1)
        Num = np.size(sortedData,0)
        
        if size%2 == 0:
            halfsize = int(size/2)
        else:
            halfsize = int((size + 1)/2)      
    
#    amplitudeScaler = config.getfloat('scan','amplitudeScaler')
#    fig = plt.figure()
#    for ii in range(0,Num,40):
#        plt.plot(time[200:3400],distanceAll[ii] + amplitudeScaler*sortedData[ii,200:3400],'k')
#    fig.show()
#    saveName = projectDir + projectName + '/result/time_CF_plot.' + save_format
#    fig.savefig(saveName,dpi = dpi,format=save_format)    
    
    fig = plt.figure()   
    plt.contourf(freq, distanceAll, np.real(FFtData),cmap='jet')  
    saveName = projectDir + projectName + '/result/freq_CF_real_plot.' + save_format
    fig.show()
    fig.savefig(saveName,dpi = dpi,format=save_format)
    
#    fig = plt.figure()
#    plt.contourf(freq, distanceAll, np.imag(FFtData),cmap='jet')    
#    saveName = projectDir + projectName + '/result/freq_CF_imag_plot.' + save_format
#    fig.show()
#    fig.savefig(saveName,dpi = dpi,format=save_format)        

def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')

def DenoiseFreqData(distance, freq, FFtData):
    num = np.size(FFtData,0)
    size = np.size(FFtData,1)

    for ii in range(num):
        FFtData[ii,:] = moving_average(FFtData[ii,:],10)
    for ii in range(size):
        FFtData[:,ii] = moving_average(FFtData[:,ii],20)
    for ii in range(size):
        FFtData[:,ii] = FFtData[:,ii]/max(FFtData[:,ii])

    return [distance, freq, FFtData]

def DenoiseFCdata(velocity, freq, fc):
    num = np.size(fc,0)
    size = np.size(fc,1)
    for ii in range(num):
        fc[ii,:] = fc[ii,:]/max(abs(fc[ii,:]))
    for ii in range(size):
        fc[:,ii] = fc[:,ii]/max(abs(fc[:,ii]))
    for ii in range(num):
        fc[ii,:] = moving_average(fc[ii,:],15)
    for ii in range(size):
        fc[:,ii] = moving_average(fc[:,ii],15)
    fc = np.maximum(fc, 0)

    return [velocity, freq, fc]


def ScanH5(sorted_path):
    h5fileName = sorted_path + 'input.h5'
    velocity = np.linspace(Scan_min_velocity,Scan_max_velocity,Scan_Npoints_velocity)
    saveName = sorted_path + 'freq.npy'
    freq = np.load(saveName)
    if not os.path.exists(h5fileName):
        saveName = sorted_path + 'distance.npy'
        distance = np.load(saveName)
        saveName = sorted_path + 'freq_CF.npy'
        FFtData = np.load(saveName)
        FFtData_real = np.real(FFtData)
        FFtData_imag = np.imag(FFtData)
        size = np.size(FFtData,1)
        Num = np.size(FFtData,0)
        [distance, freq, FFtData_real] = DenoiseFreqData(distance, freq, FFtData_real)
#        [distance, freq, FFtData_imag] = DenoiseFreqData(distance, freq, FFtData_imag) 

        h5file = h5py.File(sorted_path + 'input.h5','w')
        dset = h5file.create_dataset("nr",data=Num,dtype='int')
        dset = h5file.create_dataset("nf",data=size,dtype='int')
        dset = h5file.create_dataset("nc",data=Scan_Npoints_velocity,dtype='int')
        dset = h5file.create_dataset("r",data=distance,dtype='single')
        dset = h5file.create_dataset("f",data=freq,dtype='single')
        dset = h5file.create_dataset("c",data=velocity,dtype='single')
        dset = h5file.create_dataset("uz",data=FFtData_real,dtype='single')
        dset = h5file.create_dataset("uzI",data=FFtData_imag,dtype='single')
        h5file.close()

    inputFile = h5fileName
    outputFile = sorted_path + 'output.h5'
    if not os.path.exists(outputFile):
        command = Scan_exe + ' ' + inputFile + ' ' + outputFile
        os.system(command)

    h5file = h5py.File(outputFile,'r')
    fc_real = h5file['real'][:]
    fc_imag = h5file['imag'][:]
    h5file.close()

    fc_real = fc_real.reshape([Scan_Npoints_velocity,int(maxShift/2)])
    fc_imag = fc_imag.reshape([Scan_Npoints_velocity,int(maxShift/2)])
    [velocity, freq, fc_real] = DenoiseFCdata(velocity, freq, fc_real)
    [velocity, freq, fc_imag] = DenoiseFCdata(velocity, freq, fc_imag)
    fig = plt.figure()
    plt.contourf(freq,velocity,fc_real,cmap = 'jet')
    plt.xlabel('frequency(Hz)')
    plt.ylabel('phase velocity(m/s)')
    fig.show()
    saveName = sorted_path + 'FrequencyDispersionEnergyMap_real.' + save_format
    fig.savefig(saveName,format = save_format)

    fig = plt.figure()
    plt.contourf(freq,velocity,fc_imag,cmap = 'jet')
    plt.xlabel('frequency(Hz)')
    plt.ylabel('phase velocity(m/s)')
    fig.show()   
    saveName = sorted_path + 'FrequencyDispersionEnergyMap_imag.' + save_format
#    fig.savefig(saveName,format = save_format)

def ScanTxt(sorted_path):
    ScanTxt_exe = config.get('scan','ScanTxt_exe')
    
    inputFile = sorted_path + 'fr.txt'
    velocity = np.linspace(Scan_min_velocity,Scan_max_velocity,Scan_Npoints_velocity)
    saveName = sorted_path + 'freq.npy'
    freq = np.load(saveName)
    if not os.path.exists(inputFile):
        saveName = sorted_path + 'distance.npy'
        distance = np.load(saveName)
        saveName = sorted_path + 'freq_CF.npy'
        FFtData = np.load(saveName)
        FFtData_real = np.real(FFtData)
        FFtData_imag = np.imag(FFtData)
        size = np.size(FFtData,1)
        Num = np.size(FFtData,0)
        [distance, freq, FFtData_real] = DenoiseFreqData(distance, freq, FFtData_real)
#        [distance, freq, FFtData_imag] = DenoiseFreqData(distance, freq, FFtData_imag) 
        FFtData_real = np.reshape(FFtData_real,[1,len(freq)*len(distance)])
        freq = np.reshape(freq,[1,len(freq)])
        velocity = np.reshape(velocity,[1,len(velocity)])
        distance = np.reshape(distance,[1,len(distance)])
        with open(inputFile,'w') as fp:
            fp.writelines(str(np.size(freq,1)) + ' ' + str(np.size(velocity,1)) + ' ' + str(np.size(distance,1)) + ' ' + '5.8680e-04\n')
            np.savetxt(fp,freq)
            np.savetxt(fp,velocity)
            np.savetxt(fp,distance)
            np.savetxt(fp,FFtData_real)
            
    outputFile = sorted_path + 'fc.txt'
    if not os.path.exists(outputFile):
        command = ScanTxt_exe + ' ' + inputFile + ' ' + outputFile
        os.system(command)

    
    fc_real = np.loadtxt(outputFile)
    
    [velocity, freq, fc_real] = DenoiseFCdata(velocity, freq, fc_real)
    fc_real = fc_real.T
    fig = plt.figure()
    plt.contourf(freq, velocity, fc_real, 50, vmin = 0, cmap = 'jet')
    plt.colorbar()    
    plt.xlabel('frequency(Hz)')
    plt.ylabel('phase velocity(m/s)')
    fig.show()
    saveName = sorted_path + 'FrequencyDispersionEnergyMap_real.' + save_format
    fig.savefig(saveName,format = save_format, dpi = dpi)
    
    saveName = sorted_path + 'fcEnergy.txt'
    np.savetxt(saveName,fc_real)
    saveName = sorted_path + 'velocity.txt'
    np.savetxt(saveName,velocity)
    saveName = sorted_path + 'frequency.txt'
    np.savetxt(saveName,freq)

if __name__ == '__main__':
    Par_file = "/home/zgh/Seismology/AmbientNoise/HMDVS/Par_file.conf"
    config = configparser.ConfigParser()
    config.read(Par_file, encoding="utf-8")
    NumProcesser = config.getint('all','NumProcesser')
    channel = config.get('all','channel')
    projectDir = config.get('all','projectDir')
    projectName = config.get('all','projectName')
    sac_path = config.get('all','sac_path')
    used_sta_file = config.get('all','used_sta')
    sta_pos_file = config.get('all','sta_pos')
    pre_save_path = projectDir + projectName +'/PreprocessData/'
#    pre_save_path = sac_path
    cross_save_path = projectDir + projectName +'/CrosscorrelationData/'

    resp_path = config.get('preprocess','resp_path')
    period_min = config.getfloat('preprocess','period_min')
    period_max = config.getfloat('preprocess','period_max')
    samplate_rate = config.getfloat('preprocess','samplate_rate')
    flag_resp = config.getint('preprocess','flag_resp')
    specWhiten_method = config.getint('preprocess','specWhiten_method')
    normalization_method  = config.getint('preprocess','normalization_method')

    domain = config.get('crosscorrelation','crosscorrelateDomain')
    min_crosscorrelation_day = config.getint('crosscorrelation','min_crosscorrelation_day')
    split_hour = config.getfloat('crosscorrelation','split_hour')
    overlap_part = config.getfloat('crosscorrelation','overlap_part')
    maxShift = int(np.floor(split_hour*3600*samplate_rate))
    StackMethod = config.getint('crosscorrelation','StackMethod')
    maxStackDay = config.getint('crosscorrelation','maxStackDay')
    SNR_min = config.getfloat('crosscorrelation','SNR_min')
    SNR_window = config.getfloat('crosscorrelation','SNR_window')
    min_surfaceWave = config.getfloat('crosscorrelation','min_surfaceWave')
    max_surfaceWave = config.getfloat('crosscorrelation','max_surfaceWave')

    Scan_exe = config.get('scan','Scan_exe')
    dpi = config.getint('scan','dpi')
    Scan_min_velocity = config.getfloat('scan','Scan_min_velocity')
    Scan_max_velocity = config.getfloat('scan','Scan_max_velocity')
    Scan_Npoints_velocity = config.getint('scan','Scan_Npoints_velocity')
    save_format = config.get('scan','save_format')

    if not os.path.exists(projectDir + projectName):
        os.makedirs(projectDir + projectName)
    if not os.path.exists(projectDir + projectName + '/' + Par_file.split('/')[-1]):
        shutil.copyfile(Par_file,projectDir + projectName + '/' + Par_file.split('/')[-1])
    if not os.path.exists(projectDir + projectName + '/' + used_sta_file.split('/')[-1]):
        shutil.copyfile(used_sta_file,projectDir + projectName + '/' + used_sta_file.split('/')[-1])
    
    StationUsed = GetStation(config)
    staPos = ReadStaPosFile(StationUsed)
    
    if not config.getboolean('preprocess','pre_done'):   
        if not os.path.exists(pre_save_path):
            os.makedirs(pre_save_path)
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
                               
    if not config.getboolean('scan','sorted_done'):        
        correlate_file = os.listdir(cross_save_path)
        SortCrossFunction(correlate_file)
        
    if not config.getboolean('scan','scan_done'):        
        sorted_path = projectDir + projectName + '/result/'
#        ScanH5(sorted_path)
        ScanTxt(sorted_path)

    if not config.getboolean('pick','pick_done'):
        print('this part will complete in the future')

    if not config.getboolean('inversion','inverse_done'):
        print('this part will complete in the future')
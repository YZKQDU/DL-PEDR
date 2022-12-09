from struct import unpack,pack
import numpy as np
from math import sqrt
import math
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import optimize
import scipy.optimize as opt
import sympy

subcarriers_index = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28]
zeros = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
class CSI(object):

    def __init__(self,filename,values):
        self.filename = filename
        self.value = values
        self.csi_data = self.read_csi_data()
        self.raw_phase = self.phase_caculate()
        self.amplitude = self.amplitude_caculate()
        self.unwraped_phase = self.phase_unwrap() #相位角
        self.atf_lpf = self.lpf_filter()
        self.finger_print = self.get_fingerprinting_list()
        self.amplitude_print = self.get_amplitude_list()


    @staticmethod
    def expandable_or(a,b):
        r = a | b
        low = r & 0xff
        return unpack('b',pack('B',low))[0]

    @staticmethod
    def dbinv(x):
        return 10**(x/10)

    def total_rss(self,data):
        rssi_mag = 0
        if data['rssi_a'] != 0:
            rssi_mag = rssi_mag + (data['rssi_a'])
        if data['rssi_b'] != 0:
            rssi_mag = rssi_mag + self.dbinv(data['rssi_b'])
        if data['rssi_c'] != 0:
            rssi_mag = rssi_mag + self.dbinv(data['rssi_c'])
        return 10*np.log10(rssi_mag)-44-data['agc']

    def read_csi_data(self):
        file_name = self.filename
        data_stac = []
        triangle = np.array([1,3,6])   #三角？
        #读取dat文件内容
        with open(file_name,'rb') as f:
            buff = f.read()
            count = 0
            curr = 0
            while curr < (len(buff) - 3):
                data_len = unpack('>h',buff[curr:curr + 2])[0]
                code = unpack('B',buff[curr + 2:curr + 3])[0]
                curr = curr + 3
                if code == 187:
                    data = self.read_bfree(buff[curr:])
                    perm = data['perm']
                    nrx = data['nrx']
                    csi = data['csi']
                    if sum(perm) == triangle[nrx-1]:
                        csi[:,perm-1,:] = csi[:,[x for x in range(nrx)],:]
                    csi_matrix = self.scaled_csi(data)
                    data['csi'] = csi_matrix
                    data_stac.append(data)
                    count = count +1
                curr = curr + data_len -1

        return data_stac

    def read_bfree(self,array):

        result = {}
        array = array
        timestamp_low = array[0] + (array[1] << 8) + (array[2] << 16) + (array[3] << 24)
        bf_count = array[4] + (array[5] << 8)
        nrx = array[8]
        ntx = array[9]
        rssi_a = array[10]
        rssi_b = array[11]
        rssi_c = array[12]
        noise = unpack('b',pack('B',array[13]))[0]
        agc = array[14]
        antenna_sel = array[15]
        len_sum = array[16] +(array[17] << 8)
        fake_rate_n_flags = array[18] + (array[19] << 8)
        calc_len = (30* (nrx * ntx * 8 * 2 + 3)+ 7) // 8
        payload = array[20:]
        if len_sum != calc_len:
            print("数据发现错误！")
            exit(0)

        result['timestamp_low'] = timestamp_low
        result['bf_count'] = bf_count
        result['rssi_a'] = rssi_a
        result['rssi_b'] = rssi_b
        result['rssi_c'] = rssi_c
        result['nrx'] = nrx
        result['ntx'] = ntx
        result['agc'] = agc
        result['fake_rate_n_flags'] = fake_rate_n_flags
        result['noise'] = noise

        csi = np.zeros((ntx,nrx,30),dtype = np.complex64)

        idx = 0
        for sub_idx in range(30):
            idx = idx +3
            remainder = idx % 8
            for r in range(nrx):
                for t in range(ntx):
                    real = self.expandable_or((payload[idx // 8] >> remainder),
                                              (payload[idx // 8 + 1] << (8 - remainder)))
                    img = self.expandable_or((payload[idx // 8 + 1] >> remainder),
                                              (payload[idx // 8 + 2] << (8 - remainder)))
                    csi[t,r,sub_idx] = complex(real,img)
                    idx = idx + 16

        result['csi'] = csi
        perm = np.zeros(3,dtype = np.uint32)
        perm[0] = (antenna_sel & 0x3) + 1
        perm[1] = ((antenna_sel >> 2) & 0x3) +1
        perm[2] = ((antenna_sel >> 4) & 0x3) +1
        result['perm'] = perm
        return result

    def scaled_csi(self,data):

        csi = data['csi']
        ntx = data['ntx']
        nrx = data['nrx']

        csi_sq = csi * np.conj(csi)
        csi_pwr = csi_sq.sum().real
        rssi_pwr = self.dbinv(self.total_rss(data))
        scale = rssi_pwr / (csi_pwr / 30)

        if data['noise'] == -127:
            noise = -92
        else:
            noise = data['noise']
        thermal_noise_pwr = self.dbinv(noise)

        quant_error_pwr = scale * (nrx*ntx)
        total_nois_pwr = thermal_noise_pwr + quant_error_pwr
        ret = csi *sqrt(scale / total_nois_pwr)

        if ntx == 2:
            ret = ret * sqrt(2)
        elif ntx == 3:
            ret = ret * sqrt(self.dbinv(4.5))

        return ret

    def phase_caculate(self,x=1,y=1):
        """
        输入：帧信息数据集合
        输出：相位集合
        """
        frame_phase_list = []
        for i in self.csi_data:
            csi_matrix = i['csi'][x-1][y-1]
            raw_phase = np.angle(csi_matrix)
            frame_phase_list.append(raw_phase)
        return frame_phase_list

    def amplitude_caculate(self,x=1,y=1):
        """
        输入：帧信息数据集合
        输出：幅度集合
        """
        frame_amplitude_list = []
        for i in self.csi_data:
            csi_matrix = i['csi'][x-1][y-1]
            amplitude = np.abs(csi_matrix)
            frame_amplitude_list.append(amplitude)
        return frame_amplitude_list

    def phase_unwrap(self):
        """
        输入：未解缠绕的帧的相位集合
        输出：解缠绕后的帧的相位结合
        """
        unwraped_phase_list = list()
        for i in self.raw_phase:
            unwraped_phase_list.append(np.unwrap(i))
        return unwraped_phase_list

    def lpf_filter(self):
        """
        输入：各个帧的相位集合组成的集合，以及子载波序列[-28,-26,……]
        输出：符合条件的帧的相位集合组成的集合
        """
        fg_frame = []
        frame_len = len(self.unwraped_phase)
        for i in range(frame_len):
            r_list = list()
            for j in range(29):
                r = (self.unwraped_phase[i][j+1] - self.unwraped_phase[i][j])/(
                        subcarriers_index[j+1] -subcarriers_index[j])
                r_list.append(r)
            var_value = np.var(r_list)
            if var_value < self.value:
                fg_frame.append(self.unwraped_phase[i])
        return fg_frame

    def get_fingerprinting_list(self):
        """
        输入：归一化后的帧的子载波相位集合
        输出：将λ带入后的子载波相位集合
        """
        pi = math.pi
        frame_finger_list = list()
        for i in self.atf_lpf:
            fingerprint_list = list()
            z = (i[-1]+ i[0]) /2
            #z = (i[14]+ i[15]) /2
            k = (i[-1]- i[0]) / (112 * pi)
            #k = (i[14]- i[15]) / (4 * pi)
            for j in range(30):
                e = i[j] - (2 * pi * k *subcarriers_index[j]) - z
                fingerprint_list.append(e)
            frame_finger_list.append(fingerprint_list)
        return frame_finger_list

    def get_amplitude_list(self):
        """
        输入：子载波振幅集合
        输出：去除平均值的子载波振幅集合
        """
        frame_amplitude_list= list()
        for i in self.amplitude:
            amplitude_list = list()
            sum = 0
            for k in range(len(i)):
                sum += i[k]
            avg = sum / len(i)
            for j in range(30):
                e = i[j] - avg
                amplitude_list.append(e)
            frame_amplitude_list.append(amplitude_list)
        return frame_amplitude_list


    def average_fingerprint(self):
        return np.mean(self.finger_print,axis = 0)

    def average_amplitude(self):
        return np.mean(self.amplitude,axis = 0)

    def average_phase(self):
        return np.mean(self.unwraped_phase,axis = 0)

    def get_csi(self):
        print(self.csi_data)

    def get_raw_phase(self):
        print(self.raw_phase)

    def get_unwraped_phase(self):
        print(self.unwraped_phase)

    def get_origin_num(self):
        print(len(self.atf_lpf))

    def get_finger_num(self):
        print(len(self.finger_print))

    def get_data(self):
        data_sum = dict()
        data_sum['name'] = self.filename   #split('-')[-2]
        data_sum['csi'] = self.csi_data
        data_sum['rawpha'] = self.raw_phase
        data_sum['unwrappha'] = self.unwraped_phase
        data_sum['amt'] = self.amplitude
        data_sum['filterpha'] = self.atf_lpf
        data_sum['fplist'] = self.finger_print
        data_sum['fp'] = self.average_fingerprint()
        data_sum['amplitudef'] = self.average_amplitude()
        data_sum['aplist'] = self.amplitude_print
        return data_sum

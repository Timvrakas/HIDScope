import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import math
from scipy.signal import butter, lfilter, freqz
import filter
import scipy.signal.signaltools as sigtool
import scipy.signal as signal
from numpy.random import sample
from scipy.signal import hilbert, chirp
import peakutils
from peakutils.plot import plot as pplot

Fs = 1e6

data = np.loadtxt('binary.txt')
time = data[0]
data = data[1]

zero_t = np.arange(400)/Fs
zero = np.ones(400)*-0.5

one_t = np.arange(400)/Fs
one = np.ones(400)*0.5

preamble_t = np.arange(800*4)/Fs
preamble = np.concatenate((zero,zero,zero,one,one,one,zero,one))

bit_t = np.arange(800)/Fs
bit = np.concatenate((one,zero))

mask_t = np.arange(800*44)/Fs
mask_l = 800*44

scan = np.convolve(data,preamble[::-1],mode='same')/800
ind = peakutils.indexes(scan, thres=0.75, min_dist=2000)
ind = ind + 2*800

data_grid = []

for start in ind:
    sample = data[start:start+mask_l]
    deriv = np.diff(sample)
    plt.plot(mask_t[1:]+time[start],deriv)
    maxes = peakutils.indexes(deriv, thres=0.75, min_dist=250)
    mins = peakutils.indexes(-deriv, thres=0.75, min_dist=250)
    maxes = list(maxes)
    mins = list(mins)
    bits = str()
    t = 300
    while (t<mask_l):
        if t in maxes:
            bits += '0'
            plt.scatter(mask_t[t]+time[start],deriv[t])
            t += 650
            continue
        if t in mins:
            bits += '1'
            plt.scatter(mask_t[t]+time[start],deriv[t])
            t += 650
            continue
        t += 1
    data_grid.append(bits)


plt.plot(time,data)
#plt.plot(time,data)
#plt.plot(preamble_t+t_peak[0] + offset, preamble)
#plt.plot(preamble_t+t_peak[1] + offset, preamble)
#plt.plot(preamble_t+t_peak[2] + offset, preamble)
pplot(time, scan, ind)
plt.show()

print(data_grid[0])
for data in data_grid:
    if data != data_grid[0]:
        print(data)

import csv
with open('data1.csv', 'w') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(data_grid)

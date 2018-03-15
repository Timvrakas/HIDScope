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


def read_datafile(file_name):
    # the skiprows keyword is for heading, but I don't know if trailing lines
    # can be specified
    data = np.loadtxt(file_name, delimiter=',', skiprows=1)
    return data

def fft(signal, Fs):
    n = len(signal)  # length of the signal
    k = np.arange(n)
    T = n/Fs  # time of signal
    frq = k/T  # two sides frequency range
    frq = frq[range(math.floor(n/2))]  # one side frequency range
    Y = np.fft.fft(signal)/n  # fft computing and normalization
    Y = Y[range(math.floor(n/2))]
    return frq, Y


data = read_datafile('1.csv')

x = data[:,0]
y = data[:,1]

x = x[::10] #Downscale
y = y[::10]

Fc = 124999.3015 # Carrier Freq
Fs = 1e6

cosine = np.cos(x*Fc*2*np.pi)

demodulated = cosine*y
filtered = filter.butter_bandpass_filter(demodulated, 8000, 17000, Fs, order=4)

frqY, Y = fft(y,Fs)
frqD, D = fft(demodulated,Fs)
frqF, F = fft(filtered,Fs)



fig, ax = plt.subplots(5, 1)

ax[0].plot(frqY,abs(Y),'r') # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[0].axis([0, 30e4, 0, 0.0007])

ax[1].plot(frqD,abs(D),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|D(freq)|')
ax[1].axis([0, 30e4, 0, 0.0007])

ax[2].plot(x, demodulated)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Amplitude')
ax[2].axis([-1e-4, 1e-4, -0.11, 0.11])

ax[3].plot(frqF,abs(F),'r') # plotting the spectrum
ax[3].set_xlabel('Freq (Hz)')
ax[3].set_ylabel('|F(freq)|')
ax[3].axis([0, 30e3, 0, 0.0007])

ax[4].plot(x, filtered)
ax[4].set_xlabel('Time')
ax[4].set_ylabel('Amplitude')
ax[4].axis([-2e-3, 2e-3, -0.005, 0.005])


plt.show()

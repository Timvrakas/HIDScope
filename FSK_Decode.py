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


def fft(signal, Fs):
    n = len(signal)  # length of the signal
    k = np.arange(n)
    T = n/Fs  # time of signal
    frq = k/T  # two sides frequency range
    frq = frq[range(math.floor(n/2))]  # one side frequency range
    Y = np.fft.fft(signal)/n  # fft computing and normalization
    Y = Y[range(math.floor(n/2))]
    return frq, Y


data = np.loadtxt('1e6filtered.txt')
time = data[0]
data = data[1]

Fs = 1e6
window_time = np.arange(400) # hamming bits are 400us in length, thats the window
window_time = window_time/Fs

F1 = 12813
F2 = 15313

win_1 = np.cos(window_time*F1*2*np.pi)
win_2 = np.cos(window_time*F2*2*np.pi)


frq_win1, W1 = fft(win_1, Fs)
frq_win2, W2 = fft(win_2, Fs)
frq_data, D = fft(data,Fs)


Signal1 = np.convolve(data,win_1,mode='same')
Signal2 = np.convolve(data,win_2,mode='same')

frq_Sig1, C1 = fft(Signal1,Fs)
frq_Sig2, C2 = fft(Signal2,Fs)

Env1 = np.abs(hilbert(Signal1))
Env2 = np.abs(hilbert(Signal2))

Env = Env1-Env2

Env[Env > 0] = 0.5
Env[Env < 0] = -0.5

fig, ax = plt.subplots(3, 1)

ax[0].set_title("FSK Signal Recovery")

ax[0].plot(frq_win1, abs(W1))  # plotting the spectrum
ax[0].plot(frq_win2, abs(W2))  # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|F(freq)|')
ax[0].axis([5e3, 25e3, 0, 0.5])

ax[1].plot(frq_data,abs(D)) # plotting the spectrum
ax[1].plot(frq_Sig1,abs(C1)) # plotting the spectrum
ax[1].plot(frq_Sig2,abs(C2)) # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|S(freq)|')
ax[1].axis([5e3, 25e3, 0, 0.05])

ax[2].plot(time, Signal1)
ax[2].plot(time, Signal2)
ax[2].plot(time, Env,'black')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Recovered Signal')
ax[2].grid()
ax[2].axis([-1e-2, 1e-2, -0.6, 0.6])

plt.show()




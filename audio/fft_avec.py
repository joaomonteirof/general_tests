from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from maracas.utils import wavread

x, fs = wavread('Devel_01.wav')

b, a = signal.butter(5, 2*2000/fs)

x_new = x[45000:90000]

X = np.fft.fft(x_new)

x_filt = signal.lfilter(b, a, x_new)

X_filt = np.fft.fft(x_filt)

#plt.plot(X.real)
plt.plot(X_filt.real)
plt.show()

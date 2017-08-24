from scipy.fftpack import fft, ifft
from scipy import signal
from maracas.utils import wavread
from maracas import add_reverb

import numpy as np
import matplotlib.pyplot as plt

x, fs = wavread('sp10.wav')
assert np.allclose(fft(ifft(x)), x, atol=1e-12)

f, t, Sxx = signal.spectrogram(x, fs)

print('Shapes: {} and {}'.format(x.shape, (f.shape, t.shape, Sxx.shape)))

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

r, _ = wavread('rir.wav')

n_x = add_reverb(x, r, fs, speech_energy='P.56')

f, t, Sxx = signal.spectrogram(n_x, fs)

print('Shapes: {} and {}'.format(x.shape, (f.shape, t.shape, Sxx.shape)))

plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from maracas.utils import wavread
from scipy.io import wavfile

x, fs = wavread('Devel_01.wav')

#fs, x = wavfile.read('Devel_01.wav')

x_new = x[0:(fs*10+1)]

f, t, Sxx = signal.spectrogram(x_new, fs, scaling = 'spectrum', axis = 0, mode = 'magnitude')

print(np.max(Sxx))
print(np.min(Sxx))


Sxx_db = 20 * np.log10 ( Sxx )
print(Sxx_db.shape)

plt.pcolormesh(t, f, Sxx_db)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

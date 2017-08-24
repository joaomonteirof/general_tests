from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from maracas.utils import wavread
from scipy.io import wavfile

fs, x = wavfile.read('Devel_01.wav')

nfft = 256

pxx, freqs, bins, im = plt.specgram(x, nfft, fs)

plt.show()

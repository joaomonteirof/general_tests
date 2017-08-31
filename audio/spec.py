import numpy as np
from numpy.lib.stride_tricks import as_strided
from maracas.maracas import asl_meter
from maracas.utils import wavread
import librosa
import matplotlib.pyplot as plt

def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128, pad=0):
	"""
	Compute the spectrogram for a real signal.
	The parameters follow the naming convention of
	matplotlib.mlab.specgram

	Args:
		samples (1D array): input audio signal
		fft_length (int): number of elements in fft window
		sample_rate (scalar): sample rate
		hop_length (int): hop length (relative offset between neighboring fft windows).

	Returns:
		x (2D array): spectrogram [frequency x time]
		freq (1D array): frequency of each row in x

	Note:
	This is a truncating computation e.g. if fft_length=10,	hop_length=5 and the signal has 23 elements, then the last 3 elements will be truncated.
	"""
	assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

	window = np.hanning(fft_length)[:, None]
	window_norm = np.sum(window**2)

	# The scaling below follows the convention of
	# matplotlib.mlab.specgram which is the same as
	# matlabs specgram.
	scale = window_norm * sample_rate

	trunc = (len(samples) - fft_length) % hop_length
	x = samples[:len(samples) - trunc]

	# "stride trick" reshape to include overlap
	nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
	nstrides = (x.strides[0], x.strides[0] * hop_length)
	x = as_strided(x, shape=nshape, strides=nstrides)

	# window stride sanity check
	assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

	# broadcast window, compute fft over columns and square mod
	x = np.fft.rfft(x * window, axis=0)
	phase = np.angle(x)
	x = np.absolute(x)**2

	# scale, 2.0 for everything except dc and fft_length/2
	x[1:-1, :] *= (2.0 / scale)
	x[(0, -1), :] /= scale

	freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

	if pad > 0:
		x = np.pad(x, ((0, 0), (pad, pad)), 'constant')
		phase = np.pad(phase, ((0, 0), (pad, pad)), 'constant')

	return x, phase, freqs

def normalize(y_hat, fs, level=-26.0):
	# Normalize energy
	y_hat = y_hat/10**(asl_meter(y_hat, fs)/20) * 10**(level/20)
	return y_hat

def strided_app_givenrows(a, L, nrows):
	S = (len(a)-L) // (nrows - 1)
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

def spectrogram_from_file(filename, downsample=True, step=10, window=20, max_freq=None, eps=1e-14, log=True, normalization=True, pad=0):
	""" Calculate the log of linear spectrogram from FFT energy
	Params:
		filename (str): Path to the audio file
		step (int): Step size in milliseconds between windows
		window (int): FFT window size in milliseconds
		max_freq (int): Only FFT bins corresponding to frequencies between [0, max_freq] are returned
		eps (float): Small value to ensure numerical stability (for ln(x))
	"""

	audio, sample_rate = librosa.load(filename)

	if downsample:
		audio = librosa.resample(audio, sample_rate, 16000)
		sample_rate = 16000

	if audio.ndim >= 2:
		audio = np.mean(audio, 1)
	if max_freq is None:
		max_freq = sample_rate / 2
	if max_freq > sample_rate / 2:
		raise ValueError("max_freq must not be greater than half of sample rate")
	if step > window:
		raise ValueError("step size must not be greater than window size")

	if normalization:
		x_norm = normalize(audio, sample_rate)

	hop_length = int(0.001 * step * sample_rate)
	fft_length = int(0.001 * window * sample_rate)
	pxx, phase, freqs = spectrogram(audio, fft_length=fft_length, sample_rate=sample_rate,hop_length=hop_length, pad=pad)
	ind = np.where(freqs <= max_freq)[0][-1] + 1

	mag = pxx[:ind, :]
	melfb = librosa.filters.mel(sample_rate,fft_length,n_mels=64)
	mag = np.dot(melfb,mag)

	if log:
		return np.log(mag + eps), phase[:ind, :]
	else:
		return mag + eps, phase[:ind, :]

def spec_extraction_scipy(filename, n_outputs, stft_window=50, stft_step=25, spec_window=256, spec_height=512):

	mag, phase = spectrogram_from_file_scipy(filename, step=stft_step, window=stft_window, log=True)

	L = mag.shape[1]

	indexes = strided_app_givenrows(np.arange(L), spec_window, n_outputs)

	specs_list=[]

	for row in indexes:

		spec = mag[0:spec_height, row]
		spec.shape = (1, spec_height, len(row))
		specs_list.append(spec)

	return np.asarray(specs_list)

if __name__ == '__main__':

	file_ = '/home/joaomonteirof/Desktop/emot_mg/data/audio/Devel_01.wav'
	file_t = 'sp10.wav'


	mag, phase = spectrogram_from_file(file_, downsample = False, step=10, window=20, log=True)

	print(mag.shape)

	plt.pcolormesh(mag[:,100:132])
	plt.show()

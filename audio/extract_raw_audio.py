import numpy as np
from numpy.lib.stride_tricks import as_strided
from audio_utils import normalize
from maracas.utils import wavread
import soundfile

def read_audio(filename, scipy_=True, normalization=True):

	if scipy_:
		audio, sample_rate = wavread(filename)
		audio = audio.astype('float32')
		if audio.ndim >= 2:
			audio = np.mean(audio, 1)
	else:
		with soundfile.SoundFile(filename) as sound_file:
			audio = sound_file.read(dtype='float32')
			sample_rate = sound_file.samplerate

	if audio.ndim >= 2:
		audio = np.mean(audio, 1)

	if normalization:
		audio = normalize(audio, sample_rate)

	return audio, sample_rate

def strided_app_givenrows(a, L, nrows):
	S = (len(a)-L) // (nrows - 1)
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

def raw_audio_extraction(filename, n_outputs, length=80):

	audio, fs = read_audio(filename)

	window = int(fs * length/1000.0 )

	sliced_audio = strided_app_givenrows(audio, window, n_outputs)

	return sliced_audio

def raw_audio_extraction_scipy(filename, n_outputs, length=80):

	audio, fs = read_audio(filename, scipy_=False)

	print(audio.shape)
	print(fs)

	window = int(fs * length/1000.0 )

	sliced_audio = strided_app_givenrows(audio, window, n_outputs)

	sliced_audio = sliced_audio.reshape([sliced_audio.shape[0], 1, sliced_audio.shape[1]])

	return sliced_audio

if __name__ == '__main__':

	a = raw_audio_extraction_scipy('sp10.wav', n_outputs=100)
	print(a.shape)
	print(type(a))


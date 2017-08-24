import numpy as np
import skvideo.io
from skimage.color import rgb2gray
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt

def strided_app(a, L, S):
	nrows = ( (len(a)-L) // S ) + 1
	n = a.strides[0]
	return as_strided(a, shape=(nrows, L), strides=(S*n,n))

def rescale(data, eps = 1e-10):
	min_ = np.min(data, axis=0)
	max_ = np.max(data, axis=0)
	return (2.*(data-min_)/(max_-min_ + eps)-1.)

def rescale_labels(data, eps = 1e-10):
	min_ = np.min(data, axis=0)
	max_ = np.max(data, axis=0)
	return ((data-min_)/(max_-min_ + eps))

def normalize(data, eps = 1e-10):
	mean_ = np.mean(data, axis=0)
	std_ = np.std(data, axis=0)
	return ((data-mean_)/(std_ + eps))

def frames_extraction_DQNlike(file_name, width=224, height=224, skip_y=120, step=4):

	videogen=skvideo.io.vreader(file_name)

	frames_list = []

	for frame in videogen:

		frame = rgb2gray(frame)

		frame_shape = frame.shape

		if frame_shape[0] % 2 == 0:
			x_lowerbound = (frame_shape[1] - width)/2
			x_upperbound = (frame_shape[1] + width)/2
		else:
			x_lowerbound = (frame_shape[1]-1 - width)/2
			x_upperbound = (frame_shape[1]-1 + width)/2 + 1

		y_lowerbound = frame_shape[0] - height - skip_y
		y_upperbound = frame_shape[0] - skip_y

		frame = frame[int(y_lowerbound):int(y_upperbound), int(x_lowerbound):int(x_upperbound)]

		frames_list.append(frame)

	frames = np.asarray(frames_list)

	print(frames.shape)

	indexes = strided_app(np.arange(frames.shape[0]), 4, 1)

	DQN_frames = []

	for row in indexes:
		DQN_frames.append(frames[row,:,:])

	return np.asarray(DQN_frames)

if __name__ == '__main__':

	a = frames_extraction_DQNlike('./SampleVideo.mp4')

	print(a.shape)
	print(type(a))
	
	plt.imshow(a[58,2,:,:], interpolation='nearest')
	plt.show()

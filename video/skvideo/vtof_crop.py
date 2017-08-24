import skvideo.io
import skimage.io
import numpy as np
from skimage.color import rgb2gray


videogen=skvideo.io.vreader('Train_07.avi')

counter = 0
width = 224
height = 224
skip_y = 120

gray = True

for frame in videogen:

	counter +=1

	if gray:
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

	#frame = frame.reshape((3, int(frame.shape[0]), int(frame.shape[1])))

	if gray:
		frame = frame[int(y_lowerbound):int(y_upperbound), int(x_lowerbound):int(x_upperbound)]
	else:
		frame = frame[int(y_lowerbound):int(y_upperbound), int(x_lowerbound):int(x_upperbound), :]

	if counter>86:
		skimage.io.imsave('out/'+str(counter)+'.png', frame)
	if counter>150:
		print(frame.shape)
		break

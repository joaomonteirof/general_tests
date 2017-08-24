import skvideo.io
import skimage.io
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
import glob


def rescale(data, eps = 1e-10):
	min_ = np.min(data, axis=0)
	max_ = np.max(data, axis=0)
	return (2.*(data-min_)/(max_-min_ + eps)-1.)

def normalize(data, eps = 1e-10):
	mean_ = np.mean(data, axis=0)
	std_ = np.std(data, axis=0)
	return ((data-mean_)/(std_ + eps))

rarely = lambda aug: iaa.Sometimes(0.1, aug)
sometimes = lambda aug: iaa.Sometimes(0.25, aug)
often = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential([
		often(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.2), per_channel=0.5)), # add gaussian noise to images
		often(iaa.Dropout((0.0, 0.1), per_channel=0.5)) # randomly remove up to 10% of the pixels
	],
	random_order=True # do all of the above in random order
)

videogen=skvideo.io.vreader('Train_07.avi')

counter = 0
width = 224
height = 224
skip_y = 120

for frame in videogen:

	counter +=1

	frame_shape = frame.shape

	if frame_shape[0] % 2 == 0:
		x_lowerbound = (frame_shape[1] - width)/2
		x_upperbound = (frame_shape[1] + width)/2
	else:
		x_lowerbound = (frame_shape[1]-1 - width)/2
		x_upperbound = (frame_shape[1]-1 + width)/2 + 1

	y_lowerbound = frame_shape[0] - height - skip_y
	y_upperbound = frame_shape[0] - skip_y

	frame = frame[int(y_lowerbound):int(y_upperbound), int(x_lowerbound):int(x_upperbound), :]

	frame = frame.reshape((3, int(frame.shape[0]), int(frame.shape[1])))
	frame = frame.reshape((int(frame.shape[1]), int(frame.shape[1]), 3))

	if counter>86:
		skimage.io.imsave('out/'+str(counter)+'.png', frame)
	if counter>150:
		print(frame.shape)
		break

print('images saved')

images = glob.glob('./out/*.png')

frames_list = []

for frame in images:

	ind_image = skimage.io.imread(frame)
	ind_image = ind_image.reshape((3, int(ind_image.shape[0]), int(ind_image.shape[1])))
	ind_image = ind_image/255.0
	frames_list.append(ind_image)

frames_list = np.asarray(frames_list)

frames_list = rescale(normalize(frames_list))

for i,img in enumerate(frames_list):
	img = img.reshape((int(img.shape[1]), int(img.shape[2]), 3))
	skimage.io.imsave('out_aug/'+str(i)+'.png', img)

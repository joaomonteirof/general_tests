import skvideo.io
import skimage.io
import numpy as np
from skimage.color import rgb2gray
import dlib

out_shape = 96

detector = dlib.get_frontal_face_detector()

videogen=skvideo.io.vreader('/home/joaomonteirof/Desktop/emot_mg/data/video/Train_07.avi')

for i,frame in enumerate(videogen):

	if i>86:

		frame = rgb2gray(frame)

		im_gray = rgb2gray(frame)
		im_gray_int = np.asarray(im_gray*255.,dtype=np.uint8)
		im_v, im_h = im_gray.shape

		rect = detector(im_gray_int,1)

		try:
			sup_v = rect[0].top()+(rect[0].bottom()-rect[0].top())//2
			sup_h = rect[0].left()+(rect[0].right()-rect[0].left())//2

			top = sup_v-out_shape//2
			bottom = sup_v+out_shape//2
			left = sup_h-out_shape//2
			right = sup_h+out_shape//2

		except IndexError:
			pass

		if top<0:
			top=0
			bottom=out_shape
		elif bottom>im_v:
			bottom=im_v
			top=im_v-out_shape

		if left<0:
			left=0
			right=out_shape
		elif right>im_h:
			right=im_h
			left=im_h-out_shape


		im_cropped = im_gray[top:bottom, left:right]

		skimage.io.imsave('out/'+str(i)+'.png', im_cropped)

	if i>150:
		print(frame.shape)
		break

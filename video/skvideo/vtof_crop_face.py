import skvideo.io
import skimage.io
import numpy as np
from skimage.color import rgb2gray
import dlib
import json

out_shape = 96

detector = dlib.get_frontal_face_detector()
video_file='/home/joaomonteirof/Desktop/emot_mg/data/video/Train_07.avi'
videogen=skvideo.io.vreader(video_file)
metadata=skvideo.io.ffprobe(video_file)
print(metadata.keys())
print(json.dumps(metadata['video']))

for i,frame in enumerate(videogen):

	if i==0:
		im_v, im_h, _ = frame.shape
		c_v, c_h = im_v//2, im_h//2
		top = c_v-out_shape//2
		bottom = c_v+out_shape//2
		left = c_h-out_shape//2
		right = c_h+out_shape//2

	if i>0:

		frame = rgb2gray(frame)

		im_gray = rgb2gray(frame)
		im_gray_int = np.asarray(im_gray*255.,dtype=np.uint8)

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

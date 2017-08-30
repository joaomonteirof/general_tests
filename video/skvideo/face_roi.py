import numpy as np
import skimage.io
from skimage.color import rgb2gray
import dlib

out_shape = 64

detector = dlib.get_frontal_face_detector()

im = skimage.io.imread('testim.jpg')
im_gray = rgb2gray(im)
im_gray_int = np.asarray(im_gray*255.,dtype=np.uint8)
im_h, im_w = im_gray.shape

print(im_h, im_w)

rect = detector(im_gray_int,1)

sup_v = rect[0].top()+(rect[0].bottom()-rect[0].top())//2
sup_h = rect[0].left()+(rect[0].right()-rect[0].left())//2

print(sup_v,sup_h)

top = sup_v-out_shape//2
bottom = sup_v+out_shape//2
left = sup_h-out_shape//2
right = sup_h+out_shape//2

if top<0:
	top=0
	bottom=out_shape
elif bottom>im_h:
	bottom=im_h
	top=im_h-out_shape

if left<0:
	left=0
	right=out_shape
elif right>im_w:
	right=im_w
	left=im_w-out_shape

print(top,bottom,left,right)
print(rect[0].top(),rect[0].bottom(),rect[0].left(),rect[0].right())

print(top-bottom,left-right)
print(rect[0].top()-rect[0].bottom(),rect[0].left()-rect[0].right())

#im_cropped = im_gray[rect[0].top():rect[0].bottom(), rect[0].left():rect[0].right()]
im_cropped = im_gray[top:bottom, left:right]

print(im_gray.shape)
print(im_cropped.shape)

im = skimage.io.imsave('testim_face.jpg', im_cropped)

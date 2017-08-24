import cv2
import numpy as np
import os

lw = 0.1
lh = 0.6
hh = 0.3

img = cv2.imread('testim.jpg')
#size = (img.shape[0],img.shape[1])
size = (img.shape[1],img.shape[0])
img = cv2.resize(img,size)
lower_bounds = (int(lh*size[0]), int(lw*size[1]))
upper_bounds = (int(size[0]*(lh+hh)), int(size[1]*(1-lw)))

crop_img = img[lower_bounds[0]:upper_bounds[0],lower_bounds[1]:upper_bounds[1]]

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.imshow('crop_img',crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

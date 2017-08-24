import skvideo.io
import skvideo.utils
import skimage.io
import numpy as np
import os

def frames_to_video(inputpath,outputpath,fps):
	image_array = []
	files = [f for f in os.listdir(inputpath) if os.path.isfile(inputpath+f)]
	files.sort()
	for frame in files:
		img = skimage.io.imread(inputpath + frame)
		image_array.append(img)
	video = skvideo.utils.vshape(image_array)
	print(video.shape)
	skvideo.io.vwrite(outputpath, video)


inputpath = 'out/'
outpath =  'out.mp4'
fps = 29
frames_to_video(inputpath,outpath,fps)

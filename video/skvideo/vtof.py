import skvideo.io
import skimage.io
import numpy as np


videogen=skvideo.io.vreader('SampleVideo.mp4')

counter = 0

for frame in videogen:
	
	skimage.io.imsave('out/'+str(counter)+'.png', frame)


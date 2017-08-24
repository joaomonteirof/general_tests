import os
folder = 'frames_crop'  
os.mkdir(folder)
# use opencv to do the job
import cv2
print(cv2.__version__)  # my version is 3.1.0
vidcap = cv2.VideoCapture('SampleVideo.mp4')
count = 0
while True:
	success,image = vidcap.read()

	try:
		crop_image = image[10:400, 10:400]
	except TypeError:
		print(count)
		cv2.imshow('err', crop_image)
		cv2.waitKey(0)
	if not success:
		break
	cv2.imwrite(os.path.join(folder,"frame{:d}.jpg".format(count)), crop_image)     # save frame as JPEG file
	count += 1
print("{} images are extacted in {}.".format(count,folder))

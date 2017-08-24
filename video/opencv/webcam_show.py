import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description='Webcam test')
parser.add_argument('--gray', action='store_true', default=False, help='Grayscale')
args=parser.parse_args()

print(args.gray)

cap = cv2.VideoCapture(0)

while(True):

	ret, frame = cap.read()

	print(ret)

	if args.gray:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	cv2.imshow('frame', frame)

	if cv2.waitkey(1) & 0xFF == ord('q'):
		break

ca.release()
cv2.destroyAllWindows()

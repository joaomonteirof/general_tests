import numpy as np
import csv

def labels_extraction(filename):

	to_return = np.empty([1,3])

	with open(filename, newline='') as f:

		reader = csv.reader(f, delimiter=';')

		for row in reader:
			new_row = np.asarray( [ float(row[2]), float(row[3]), float(row[4]) ] )

			to_return = np.vstack( ( to_return, new_row ) )
	
	to_return = to_return[1:,:]

	return to_return

def rescale_labels(data, eps = 1e-10):
	min_ = np.min(data, axis=0)
	max_ = np.max(data, axis=0)
	return ((data-min_)/(max_-min_ + eps))

if __name__ == '__main__':

	a = labels_extraction('/home/joaomonteirof/Desktop/emot/data/labels/Devel_01.csv')

	print(a.shape)
	print(a[:30])

	a = rescale_labels(a)

	print(a.shape)
	print(a[:30])

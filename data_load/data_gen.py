import numpy as np

def myGenerator(batch_size = 2):

	a=[1,2,3,4,5,6,7,8,9,10]
	data_size = len(a)
	number_of_batches = int(np.ceil(data_size/batch_size))

	for i in xrange(0, number_of_batches):
		inputs_batch = a[i*batch_size:min((i+1)*batch_size, data_size)]

		yield inputs_batch

if __name__ == '__main__':

	gen = myGenerator()
	for i, data in enumerate(gen):
		print(i)
		print(data)

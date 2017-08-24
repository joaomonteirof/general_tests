import numpy as np
import glob
import pickle

class my_gen():

	def __init__(self, batch_size):
		self.batch_size = batch_size

	def generate(self):

		files_list = glob.glob('./*.p')

		for ind_file in files_list:
			arr = pickle.load(open(ind_file,'rb'))
			print(ind_file)
			i = 0

			while True:

				a = arr[i*batch_size:(batch_size+i*batch_size)]

				yield a

				if len(a)<self.batch_size:
					break
				print(i*self.batch_size, (self.batch_size+i*self.batch_size))
				print(len(a), self.batch_size)
				i += 1
		

if __name__ == '__main__':

	batch_size = 5
	gen = my_gen(batch_size)

	print('1st')

	for i in gen.generate():
		print(i)

	print('2nd')

	for i in gen.generate():
		print(i)

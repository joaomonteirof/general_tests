import numpy as np
from tqdm import tqdm
import torch

def generate(arr, batch_size):

	i = 0

	while True:

		a = arr[i*batch_size:(batch_size+i*batch_size)]

		yield a[0:]

		if a.shape[0]<batch_size:
			break
		print(i*batch_size, (batch_size+i*batch_size))
		print(a.shape[0], batch_size)
		i += 1

class my_gen():

	def __init__(self, arr, batch_size):
		self.arr = arr
		self.batch_size = batch_size

	def generate(self, *args):

		i = 0

		while True:

			a = self.arr[i*batch_size:(batch_size+i*batch_size)]

			if a.shape[0]>0:

				yield a[0:]

			if a.shape[0]<self.batch_size:
				break
			print(i*self.batch_size, (self.batch_size+i*self.batch_size))
			print(a.shape[0], self.batch_size)
			i += 1

class my_gen_tensor():

	def __init__(self, arr, batch_size):
		self.arr = arr
		self.batch_size = batch_size

	def generate(self, *args):

		i = 0

		while True:

			a = self.arr[i*batch_size:(batch_size+i*batch_size)]

			try:
				a = torch.from_numpy(a)
				current_size = a.size()[0]
				yield a[0:]

			except RuntimeError:
				break

			if current_size<self.batch_size:
				break

			print(i*self.batch_size, (self.batch_size+i*self.batch_size))
			print(a.size()[0], self.batch_size)
			i += 1
		

if __name__ == '__main__':

	a = np.random.random([27,4,5,6])
	print(a.shape)

	batch_size = 9

	gen = my_gen(a,batch_size)

	print('1st')

	iter_ = tqdm(enumerate(gen.generate()))

	for i, j in iter_:
		print(i, j.shape)

	print('2nd')

	iter_ = tqdm(enumerate(gen.generate()))

	for i, j in iter_:
		print(i, j.shape)

	gen = my_gen_tensor(a,batch_size)

	print('3rd')

	iter_ = tqdm(enumerate(gen.generate()))

	for i, j in iter_:
		print(i, j.size())

	print('4th')

	iter_ = tqdm(enumerate(gen.generate()))

	for i, j in iter_:
		print(i, j.size())

from multiprocessing import Pool
import psutil
from pathos.multiprocessing import ProcessingPool
from scoop import futures

def incrementer(el):
	el += 1.
	return el

if __name__ == '__main__':

	a = [1., 2., 3., 4.]

	b = a

	c = a

	d = a

	print(a)
	print(b)
	print(c)
	print(d)


	a = list(map(incrementer, a))

	print(a)
	print(b)
	print(c)
	print(d)

	pool = Pool(psutil.cpu_count(logical=False))

	b = list(pool.map(incrementer, b))

	print(a)
	print(b)
	print(c)
	print(d)

	c = ProcessingPool().map(incrementer, c)

	print(a)
	print(b)
	print(c)
	print(d)

	d = futures.map(incrementer, d)

	print(a)
	print(b)
	print(c)
	print(d)

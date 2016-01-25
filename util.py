from timeit import default_timer as timer

N = 10**8
runs=10
mpi_verbose=False

def timeit(func, name, count=runs):
	"""
	reimplementation of timeit with return value
	"""
	start = timer()
	for i in range(count):
		ret = func()
	end = timer()
	print("Runtime of " + name + ": " + str((end - start)/count) + "s")
	print("Result of " + name + ": " + str(ret))
	return ret

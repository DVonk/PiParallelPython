from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl
from pyopencl import array as clarray
import math
import time
import os

N = 50000000

def get_opencl(verbose=False):
	"""
	Returns an opencl context and queue.
	"""
	platform = cl.get_platforms()
	my_gpu_devices = platform[0].get_devices()	#cl.device_type.GPU
	if verbose:
		print("Platforms: " + str(platform))
		print("OpenCL devices: " + str(my_gpu_devices))
	ctx = cl.Context([my_gpu_devices[0]])
	queue = cl.CommandQueue(ctx)
	return ctx, queue

def calculate_pi_opencl(n, cores=0):
	"""
	Calculate pi in n cycles with opencl on specified number of cores. cores=0 => all cores
	"""
	ctx, queue = get_opencl()
	prg = cl.Program(ctx, """
	__kernel void pi(__global float *res_g, const int n) {
	  int gid = get_global_id(0);
	  float x = (((float) gid - 0.5)/(float) n);
	  res_g[gid] = 4/(1 + x*x);
	}
	""").build()
	mf = cl.mem_flags
	res_np = np.zeros(n, dtype=np.float32)
	res_g = cl.Buffer(ctx, mf.WRITE_ONLY, res_np.nbytes)
	prg.pi(queue, res_np.shape, None, res_g, np.int32(n))
	cl.enqueue_copy(queue, res_np, res_g)
	#print("res: " + str(res_np.sum()/n))
	return res_np.sum()/n

def calculate_pi_opencl_simple(n, cores=0):
	"""
	Calculate pi in n cycles with opencl on specified number of cores. cores=0 => all cores
	This version uses the build-in map reduce of pyopencl.
	"""
	#os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
	ctx, queue = get_opencl()
	res = clarray.zeros(queue, n, dtype=np.float32)
	from pyopencl.reduction import ReductionKernel
	pi_reduce = ReductionKernel(ctx, np.float32, neutral="0",
			reduce_expr="a+b", map_expr="4.0/(1.0 + (((float) i - 0.5)/n)*(((float) i - 0.5)/N))",
			arguments="__global float *res, float N")
	pi_n = pi_reduce(res, np.float32(N)).get()
	#print("res simple: " + str(pi_n/N))
	return pi_n/N

def timeit(func, n, count):
	"""
	reimplementation of timeit with return value
	"""
	start = time.clock()
	for i in range(count):
		ret = func(n)
	end = time.clock()
	print("Runtime of " + func.__name__ + str((end - start)/count) + "s")
	print("Result of " + func.__name__ + ": " + str(ret))
	return ret

def main():
	"""
	main
	"""
	timeit(calculate_pi_opencl, N, 5)
	timeit(calculate_pi_opencl_simple, N, 5)
	print("pi: " + str(4.0*math.atan(1)))

if __name__=='__main__':
	main()

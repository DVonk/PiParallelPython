
from subprocess import check_output
import util
import sys
import math
from util import N, timeit
from functools import partial
import multiprocessing
import pi_serial
import pi_multiprocessing
#import pi_mpi
import pi_openCL


def do_mpi(cpus):
	util.mpi_verbose = False #make sure pi_mpi.py prints only its result to console
	mpi_path = sys.path[0]
	#get result and strip trailing "\n"
	return check_output(["mpiexec -n " + str(int(cpus)) + " python ./pi_mpi.py"], shell=True)[:-2]

if __name__=='__main__':
	print("N: " + "{:,}".format(N))
	print("Pi: " + str(math.pi))
	pi_serial.run()
	pi_multiprocessing.run()
	pi_openCL.run()
	#do mpi:
	for cpus in range(multiprocessing.cpu_count()):
		timeit(partial(do_mpi, cpus+1), "MPI with " + str(cpus+1) + " CPUs")

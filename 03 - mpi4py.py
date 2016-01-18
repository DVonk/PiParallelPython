# -*- coding: utf-8 -*-

import math
from mpi4py import MPI
from timeit import default_timer as timer
    
def compute_pi(n, start=0, step=1):
    h = 1.0 / n
    s = 0.0
    for i in range(start, n, step):
        x = h * (i + 0.5)
        s += 4.0 / (1.0 + x**2)
    return s * h
    
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("Rank:", rank, "Size:", size)

if rank == 0:
    n = 10000000
else:
    n = None
    
start = timer()

n = comm.bcast(n, root=0)
mypi = compute_pi(n, rank, size)
pi = comm.reduce(mypi, op=MPI.SUM, root=0)

end = timer()


if(rank == 0):
    error = abs(pi - math.pi)
    print ("pi is app. %.16f, error is %.16f" % (pi, error))
    print ("Using", size, "CPU(s)")
    print ("Time:", end - start)
    print ("--------------------------")

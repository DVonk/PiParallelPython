# -*- coding: utf-8 -*-

import math
from mpi4py import MPI
from timeit import default_timer as timer
from util import N, mpi_verbose
    
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

if mpi_verbose:
    print("Rank: " + str(rank) + " Size: " + str(size))

if rank == 0:
    n = N
    start = timer()
else:
    n = None
    
n = comm.bcast(n, root=0)
mypi = compute_pi(n, rank, size)
pi = comm.reduce(mypi, op=MPI.SUM, root=0)

if rank == 0:
    end = timer()
    error = abs(pi - math.pi)
    if mpi_verbose:
        print("pi is app. " + str(pi) + ", error is " + str(error))
        print("Using " + str(size) + " CPU(s)")
        print("Time: " + str(end - start) + "s")
        print("--------------------------")
    else:
        print(str(pi))

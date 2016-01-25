import multiprocessing
import math
from multiprocessing import Pool
from functools import partial
from timeit import default_timer as timer

#N = 10**7
from util import N, timeit

#========== Estimate Pi (according to lecture slides) ==========
def estimate_pi(n):   
    count = 0 
    for i in range(n):
        x = ((i - 0.5) / n)
        count += 4.0 / (1 + x**2)       
    return count

#========== Parallel computing via multiprocessing map ==========
def calculate_pi(cpus, n):
    #Divide workload (n) on number of CPUs
    part_count = [int(round(n/cpus)) for i in range(cpus)]
    #Define pool with one process per CPU
    pool = Pool(processes = cpus)

    #Use map to calculate part of the solution and time it
    count = pool.map(estimate_pi, part_count)
    #Divide by n (according to lecture slides) 
    pi = sum(count)/n
    return pi

#========== do a benchmark run ==============
def run():
    for cpus in range (1, multiprocessing.cpu_count()+1):
        timeit(partial(calculate_pi, cpus, N), "multiprocessing with " + str(cpus) + " CPUs")

#========== Main ==========
if __name__=='__main__':
    #Define n and number of CPUs
    pi = 0
    cpus = multiprocessing.cpu_count()
    print("CPU count: " + str(cpus))
    print("n = " +  str(N))
    print("--------------------------")
    print("--------------------------")

    #Estimate Pi
    for i in range(1, cpus+1):
        print("Using " + str(i) + " CPU(s)")
        start = timer()
        #calc = calculate_pi(i+1, N)
        pi += timeit(partial(calculate_pi, i, N), "multiprocessing")
        end = timer()
        print("Time: " + str(end - start) + "s")
        print("--------------------------")

    print("--------------------------")
    print("Pi: " +  str(pi/cpus))
    print("Error: " +  str(abs(pi/cpus - math.pi)))

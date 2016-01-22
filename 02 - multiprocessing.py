import multiprocessing
from multiprocessing import Pool
from timeit import default_timer as timer

chunk_size = 100000
N = 100000000

#========== Estimate Pi (according to lecture slides) ==========
def estimate_pi(n):   
    count = 0 
    for i in range(chunk_size*(n-1), chunk_size*n):
        x = ((i - 0.5) / N)
        count += 4.0 / (1 + x**2)       
    return count

#========== Parallel computing via multiprocessing map ==========
def calculate_pi(cpus, n):
    #Divide workload (n) on number of CPUs
    part_count = [i for i in range(1, int(round(n/chunk_size) + 1))]
    #Define pool with one process per CPU
    pool = Pool(processes = cpus)

    #Use map to calculate part of the solution and time it
    start = timer()
    count = pool.map(estimate_pi, part_count)   
    end = timer() 

    #Print some stuff
    print("Using " + str(cpus) + " CPU(s)") 
    print("Time: " + str(end - start))
    print("--------------------------")

    #Divide by n (according to lecture slides)
    pi = sum(count)/n
    return pi

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
    for i in range (cpus):
        pi += calculate_pi(i+1, N)

    print("--------------------------")
    print("Pi: " +  str(pi/cpus))

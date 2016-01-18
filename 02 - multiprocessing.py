import multiprocessing
from multiprocessing import Pool
from timeit import default_timer as timer

#========== Estimate Pi (according to lecture slides) ==========
def estimate_pi(n):   
    count = 0 
    for i in range(n):
        x = ((i - 0.5) / n)
        count += 4.0 / (1 + x * x)       
    return count

#========== Parallel computing via multiprocessing map ==========
def calculate_pi(cpus, n):
    #Divide workload (n) on number of CPUs
    part_count = [n/cpus for i in range(cpus)]
    #Define pool with one process per CPU
    pool = Pool(processes = cpus)

    #Use map to calculate part of the solution and time it
    start = timer()
    count = pool.map(estimate_pi, part_count)   
    end = timer() 

    #Print some stuff
    print "Using", cpus, "CPU(s)"
    print "Time:", end - start
    print "--------------------------"

    #Divide by n (according to lecture slides)
    pi = sum(count)/n
    return pi

#========== Main ==========
if __name__=='__main__':

    #Define n and number of CPUs
    n = 100000000
    pi = 0
    cpus = multiprocessing.cpu_count()
    print "CPU count:", cpus
    print "n =", n
    print "--------------------------"
    print "--------------------------"

    #Estimate Pi
    for i in range (cpus):
        pi += calculate_pi(i+1, n)

    print "--------------------------"
    print "Pi: ", pi/cpus

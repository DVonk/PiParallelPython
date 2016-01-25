#from timeit import default_timer as timer

from util import N, timeit

def estimate_pi(n):   
    count = 0 
    for i in range(n):
        x = ((i - 0.5) / n)
        count += 4.0 / (1 + x * x)       
    return count

def pi_serial():
	"""Simple wrapper"""
	return estimate_pi(N)/N

def run():
	timeit(pi_serial, "serial")

if __name__=='__main__':
	run()

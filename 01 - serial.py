from timeit import default_timer as timer

def estimate_pi(n):   
    count = 0 
    for i in range(n):
        x = ((i - 0.5) / n)
        count += 4.0 / (1 + x * x)       
    return count

n = 100000000

start = timer()
pi = estimate_pi(n)/n
end = timer()

print (pi)
print "Time:", end - start

import time
st = float(open("mytime.txt","r").read())
print("Total Time: %.1f seconds"%(time.time()-st))
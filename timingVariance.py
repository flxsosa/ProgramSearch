from utilities import *

import os
import sys

# map from benchmark name to list of times
benchmarkTimes = {}

for d in sys.argv[1:]:
    for fn in os.listdir(d):
        if fn.endswith("SMC_value.pickle"):
            benchmarkTimes[fn] = benchmarkTimes.get(fn,[])
            path = d + "/" + fn
            with open(path, "rb") as handle:
                R = pickle.load(handle)
                benchmarkTimes[fn].append(R)
                print(path)
                print(R)
                print()

            
    
    

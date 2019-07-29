from utilities import *

import os
import sys


# map from benchmark name to list of times
benchmarkTimes = {}

for d in sys.argv[1:]:
    for fn in os.listdir(d):
        if fn == "testResults.pickle":
            path = d + "/" + fn
            with open(path, "rb") as handle:
                R = pickle.load(handle)[0][1]
                for ri,r in enumerate(R):
                    successfulResults = [_r for _r in r if _r.loss < 0.1 ]
                    if successfulResults:
                        benchmarkTimes[ri] = benchmarkTimes.get(ri,[]) + [successfulResults[0].time]
deviations = []
for k,ts in benchmarkTimes.items():
    print(k,ts,standardDeviation(ts))
    deviations.append(standardDeviation(ts))
print(deviations, mean(deviations))

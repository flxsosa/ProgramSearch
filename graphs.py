from CAD import *
from utilities import *

import matplotlib.pyplot as plot

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import pickle


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument("checkpoints", nargs='+')
    parser.add_argument("--timeout","-t",type=int,default=None)
    parser.add_argument("--export","-e",type=str,default=None)
    parser.add_argument("--title","-n",type=str,default=None)
    arguments = parser.parse_args()

    solverToResult = {}
    for fn in arguments.checkpoints:
        with open(fn,"rb") as handle:
            for solver, results in pickle.load(handle):
                if solver in solverToResult:
                    assert False
                solverToResult[solver] = results

    timeout = arguments.timeout
    if timeout is None:
        timeout = max(r_.time
                      for rs in solverToResult.values()
                      for r in rs
                      for r_ in r )
        print("setting timeout to",timeout)


        
    plot.figure(figsize=(4,3))
    #plot.tight_layout();
    plot.ylabel("Intersection over Union")
    plot.xlabel("Time (seconds)")
    plot.ylim([0,1])
    if arguments.title:
        plot.title(arguments.title)

    def rewardAtTime(t,solver):
        return [1 - min([r.loss for r in rs if r.time <= t] + [1.])
                for rs in solverToResult[solver] ]

    ordering = ["SMC_value","forwardSample","beam_value","beam","no_REPL"]
    name = {"SMC_value": "SMC",
            "forwardSample": "policy rollout",
            "beam_value": "beam w/ value",
            "beam": "beam w/o value",
            "no_REPL": "no REPL"}
    ordering = dict(zip(ordering,range(len(ordering))))
    for s in sorted(solverToResult.keys(),
                    key=lambda s: ordering[s]):
        ts = np.arange(0,timeout,0.1)
        ys = [rewardAtTime(t,s) for t in ts ]
        ys = [mean(y) for y in ys ]
        plot.semilogx(ts,ys,label=name[s],linewidth=3,basex=10)
    plot.legend()
        

    if arguments.export:
        plot.savefig(arguments.export)
    else:
        plot.show()

    
    


    

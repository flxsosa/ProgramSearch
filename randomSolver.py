from API import *
from programGraph import *

import time
import random

class RandomSolver(Solver):
    def __init__(self, DSL):
        self.DSL = DSL

    def _infer(self, spec, loss, timeout):
        t0 = time.time()

        g = ProgramGraph([])

        def getArgument(requestedType):
            if requestedType.isInteger:
                return random.choice(range(requestedType.lower, requestedType.upper + 1))
            
            choices = [o for o in g.objects() if requestedType.instance(o)]
            if choices: return random.choice(choices)
            else: return None

        while time.time() - t0 < timeout:

            # Pick a random DSL production
            operator = random.choice(self.DSL.operators)
            tp = operator.type

            if not tp.isArrow:
                object = operator()
            else:
                # Sample random arguments
                arguments = [getArgument(t) for t in tp.arguments]
                if any( a is None for a in arguments ): continue
                try:
                    object = operator(*arguments)
                except: continue

            if object not in g.objects():
                g = g.extend(object)
                self._report(ProgramGraph.fromRoot(object))
                

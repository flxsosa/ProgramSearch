from API import *
from programGraph import *
from pointerNetwork import *
from exit import *

import time

class BeamSearch(ExitSolver):
    def __init__(self, model, maximumLength=None, criticCoefficient=0.):
        self.criticCoefficient = criticCoefficient
        self.maximumLength = maximumLength
        self.model = model

    @property
    def name(self):
        if self.criticCoefficient > 0.:
            return "beam_value"
        else:
            return "beam"

    def _infer(self, spec, loss, timeout):

        specEncoding = self.model.specEncoder(spec.execute())

        objectEncodings = ScopeEncoding(self.model)

        B = 5
        exponentialGrowthFactor = 2

        startTime = time.time()

        model = self.model
        class Particle():
            def __init__(self, graph, ll, trajectory, finished=False, newObject=None):
                self.newObject = newObject
                self.trajectory = trajectory
                self.graph = graph
                self.ll = ll
                self.distance = None
                self.finished = finished
                self.reported = False
            def __str__(self):
                return f"Particle(ll={self.ll}, -logV={self.distance}, finished={self.finished}, graph=\n{self.graph.prettyPrint()}\n)"
        allObjects = set()

        while time.time() - startTime < timeout:
            population = [Particle(ProgramGraph([]), 0., [])]
            while any( not p.finished for p in population ):
                children = []
                for p in population:
                    if p.finished:
                        children.append(p)
                        continue

                    for o, l in self.model.beamNextLine(spec, specEncoding, p.graph, objectEncodings, B):
                        if o is None:
                            children.append(Particle(p.graph, p.ll + l, p.trajectory, finished=True))
                        else:
                            children.append(Particle(p.graph.extend(o), p.ll + l, p.trajectory + [o], newObject=o))

                newObjects = {p.newObject for p in children
                              if p.newObject is not None and p.newObject not in allObjects }
                for o in newObjects: allObjects.add(o)
                objectEncodings.registerObjects([(o,spec) for o in newObjects])

                for c in children:
                    oe = objectEncodings.encoding(spec, c.graph.objects(oneParent=model.oneParent))
                    d = model.distance(oe, specEncoding).cpu().data.item()
                    c.distance = d                


                children.sort(key=lambda p: -p.ll + p.distance*self.criticCoefficient)
                population = children[:B]


                if True:
                    print("Population (top 3):")
                    for p in population[:3]:
                        print(p)

                for p in population:
                    if p.finished and not p.reported:
                        self._report(p.graph, p.trajectory)
                        p.reported = True
                if self.maximumLength is not None and self.maximumLength <= max(len(p.graph) for p in children):
                    # print("Exiting early because we went beyond the maximum length")
                    return

            B*=exponentialGrowthFactor
            print("Increased beam size to",B)

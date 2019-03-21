from API import *
from programGraph import *
from pointerNetwork import *
from exit import *

class BeamSearch(ExitSolver):
    def __init__(self, model, maximumLength=None):
        self.maximumLength = maximumLength
        self.model = model

    def _infer(self, spec, loss, timeout):

        specEncoding = self.model.specEncoder(spec.execute())

        objectEncodings = ScopeEncoding(self.model)

        B = 20

        class Particle():
            def __init__(self, graph, ll, trajectory, finished=False, newObject=None):
                self.newObject = newObject
                self.trajectory = trajectory
                self.graph = graph
                self.ll = ll
                self.finished = finished
                self.reported = False
            def __str__(self):
                return f"Particle(ll={self.ll}, finished={self.finished}, graph=\n{self.graph.prettyPrint()}\n)"

        population = [Particle(ProgramGraph([]), 0., [])]
        allObjects = set()

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

            children.sort(key=lambda p: -p.ll)
            population = children[:B]
            
            newObjects = {p.newObject for p in population
                          if p.newObject is not None and p.newObject not in allObjects }
            for o in newObjects: allObjects.add(o)
            objectEncodings.registerObjects([(o,spec) for o in newObjects])
            
            if False:
                print("Population:")
                for p in population:
                    print(p)
                    
            for p in population:
                if p.finished and not p.reported:
                    self._report(p.graph, p.trajectory)
                    p.reported = True
            if self.maximumLength is not None and self.maximumLength <= max(len(p.graph) for p in children):
                # print("Exiting early because we went beyond the maximum length")
                return 

from API import *
from programGraph import *
from pointerNetwork import *
from exit import *

class BeamSearch(ExitSolver):
    def __init__(self, model):
        self.model = model

    def _infer(self, spec, loss, timeout):

        specEncoding = self.model.specEncoder(spec.execute())

        objectEncodings = ScopeEncoding(self.model)

        B = 5

        class Particle():
            def __init__(self, graph, ll, trajectory, finished=False):
                self.trajectory = trajectory
                self.graph = graph
                self.ll = ll
                self.finished = finished
                self.reported = False

        population = [Particle(ProgramGraph([]), 0., [])]

        while any( not p.finished for p in population ):
            children = []
            for p in population:
                if p.finished: continue
                
                for o, l in self.model.beamNextLine(spec, specEncoding, p.graph, objectEncodings, B):
                    if o is None:
                        children.append(Particle(p.graph, p.ll + l, p.trajectory, finished=True))
                    else:
                        children.append(Particle(p.graph.extend(o), p.ll + l, p.trajectory + [o]))

            children.sort(key=lambda p: -p.ll)
            population = children[:B]

            for p in children:
                if p.finished and not p.reported:
                    self._report(p.graph, p.trajectory)
                    p.reported = True

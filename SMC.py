from programGraph import *
from API import *
from pointerNetwork import *

import time

class SMC(Solver):
    def __init__(self, model, _=None,
                 maximumLength=8,
                 initialParticles=100, exponentialGrowthFactor=2,
                 fitnessWeight=2.):
        self.maximumLength = maximumLength
        self.initialParticles = initialParticles
        self.exponentialGrowthFactor = exponentialGrowthFactor
        self.fitnessWeight = fitnessWeight
        self.model = model

    def _infer(self, spec, loss, timeout):
        startTime = time.time()
        numberOfParticles = self.initialParticles
        
        specEncoding = self.model.specEncoder(spec)
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model, spec)

        # Maps from a graph to its distance
        _distance = {}
        def distance(g):
            if g in _distance: return _distance[g]
            se = objectEncodings.encoding(list(g.objects()))
            d = self.model.distance(se, specEncoding)
            _distance[g] = d
            return d            
        
        class Particle():
            def __init__(self, graph, frequency):
                self.frequency = frequency
                self.graph = graph
                self.distance = distance(graph)


        while True:
            population = [Particle(ProgramGraph([]), numberOfParticles)]
            for _ in range(self.maximumLength):
                sampleFrequency = {}
                for p in population:
                    for newObject in self.model.repeatedlySample(specEncoding, p.graph,
                                                                 objectEncodings, p.frequency):
                        if newObject is None: newGraph = p.graph
                        else: newGraph = p.graph.extend(newObject)                        
                        sampleFrequency[newGraph] = sampleFrequency.get(newGraph, 0) + 1

                if time.time() - startTime >= timeout: return

                for g in sampleFrequency: self._report(g)

                # Convert graphs to particles
                samples = [Particle(g, f)
                           for g, f in sampleFrequency.items() ]

                # Resample
                logWeights = [math.log(p.frequency) - p.distance
                              for p in samples]
                ps = [ math.exp(lw - max(logWeights)) for lw in logWeights ]
                ps = [p/sum(ps) for p in ps]
                sampleFrequencies = np.random.multinomial(numberOfParticles, ps)

                population = []
                for particle, frequency in zip(samples, sampleFrequencies):
                    if frequency > 0:
                        particle.frequency = frequency
                        population.append(particle)

            numberOfParticles *= self.exponentialGrowthFactor

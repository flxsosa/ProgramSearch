from programGraph import *
from API import *
from pointerNetwork import *

import time

class SMC(Solver):
    def __init__(self, model, _=None,
                 maximumLength=8,
                 initialParticles=100, exponentialGrowthFactor=2,
                 criticCoefficient=1.):
        self.maximumLength = maximumLength
        self.initialParticles = initialParticles
        self.exponentialGrowthFactor = exponentialGrowthFactor
        self.criticCoefficient = criticCoefficient
        self.model = model
        
    @property
    def name(self): return "SMC_value"
    
    def _infer(self, spec, loss, timeout):
        startTime = time.time()
        numberOfParticles = self.initialParticles
        
        specEncoding = self.model.specEncoder(spec.execute())
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model)
        
        allObjects = set()

        class Particle():
            def __init__(self, trajectory, frequency, finished=False):
                self.frequency = frequency
                self.trajectory = trajectory
                self.graph = ProgramGraph(trajectory)
                self.distance = None
                self.finished = finished
                self.reported = False
            def __str__(self):
                return f"Particle(frequency={self.frequency}, -logV={self.distance}, finished={self.finished}, graph=\n{self.graph.prettyPrint()}\n)"
                
            @property
            def immutableCode(self):
                return (self.graph, self.finished)
                
            def __eq__(self,o):
                return self.immutableCode == o.immutableCode
            def __ne__(self,o): return not (self == o)
            def __hash__(self): return hash(self.immutableCode)

        while time.time() - startTime < timeout:
            population = [Particle(tuple([]), numberOfParticles)]
            for _ in range(self.maximumLength):
                sampleFrequency = {} # map from (trajectory, finished) to frequency
                newObjects = set()
                for p in population:
                    for newObject in self.model.repeatedlySample(spec, specEncoding, p.graph,
                                                                 objectEncodings, p.frequency):
                        if newObject is None:
                            newKey = (p.trajectory, True)
                        else:
                            newKey = (tuple(list(p.trajectory) + [newObject]), False)
                            if newObject not in allObjects:
                                newObjects.add(newObject)
                            
                        sampleFrequency[newKey] = sampleFrequency.get(newKey, 0) + 1
                        
                for o in newObjects: allObjects.add(o)
                objectEncodings.registerObjects([(o,spec) for o in newObjects])

                for t,f in sampleFrequency:
                    if f:
                        self._report(ProgramGraph(t))

                # Convert trajectories to particles
                samples = [Particle(t, frequency, finished=finished)
                           for (t, finished), frequency in sampleFrequency.items() ]
                
                # Computed value
                for p in samples:
                    oe = objectEncodings.encoding(spec, p.graph.objects(oneParent=self.model.oneParent))
                    p.distance = self.model.distance(oe, specEncoding).cpu().data.item()

                # Resample
                logWeights = [math.log(p.frequency) - p.distance*self.criticCoefficient
                              for p in samples]
                ps = [ math.exp(lw - max(logWeights)) for lw in logWeights ]
                ps = [p/sum(ps) for p in ps]
                sampleFrequencies = np.random.multinomial(numberOfParticles, ps)

                population = []
                print("Samples:")
                for particle, frequency in sorted(zip(samples, sampleFrequencies),
                                                  key=lambda sf: sf[1]):
                    particle.frequency = frequency
                    if frequency > 0.3*numberOfParticles:
                        print(particle)
                        print()
                    if frequency > 0 and not particle.finished:
                        particle.frequency = frequency
                        population.append(particle)
                        
                if len(population) == 0: break
                
            numberOfParticles *= self.exponentialGrowthFactor
            print("Increased number of particles to",numberOfParticles)

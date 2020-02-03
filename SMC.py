from utilities import *
from programGraph import *
from API import *
from pointerNetwork import *

import os
import time

INSTRUMENT = True
def instrumentSMC(prefix):
    global INSTRUMENT
    INSTRUMENT = prefix

class SMC(Solver):
    def __init__(self, model, _=None,
                 maximumLength=8,
                 initialParticles=20, exponentialGrowthFactor=2,
                 criticCoefficient=1.):
        self.maximumLength = maximumLength
        self.initialParticles = initialParticles
        self.exponentialGrowthFactor = exponentialGrowthFactor
        self.criticCoefficient = criticCoefficient
        self.model = model
        
    @property
    def name(self): return "SMC_value"
    
    def _infer(self, spec, loss, timeout):
        global INSTRUMENT
        
        startTime = time.time()
        numberOfParticles = self.initialParticles
        
        specEncoding = self.model.specEncoder(spec.execute())
        
        # Maps from an object to its embedding
        objectEncodings = ScopeEncoding(self.model)
        
        allObjects = set()

        if INSTRUMENT:
            os.system(f"mkdir  -p experimentOutputs/SMC/{INSTRUMENT}")
            
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
            if INSTRUMENT: os.system(f"mkdir  -p experimentOutputs/SMC/{INSTRUMENT}/{numberOfParticles}")
            for generation in range(self.maximumLength):
                if time.time() - startTime > timeout: break
                
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

                if INSTRUMENT:
                    os.system(f"mkdir  -p experimentOutputs/SMC/{INSTRUMENT}/{numberOfParticles}/generation{generation}")
                
                        
                for o in newObjects: allObjects.add(o)
                objectEncodings.registerObjects([(o,spec) for o in newObjects])

                for t,f in sampleFrequency:
                    if f or True:
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
                childIndex = 0
                for particle, frequency in sorted(zip(samples, sampleFrequencies),
                                                  key=lambda sf: sf[1]):
                    if INSTRUMENT:
                        childIndex += 1
                        trajectory = '\n'.join([str(ptt) for ptt in particle.trajectory])
                        stringToFile(f"experimentOutputs/SMC/{INSTRUMENT}/{numberOfParticles}/generation{generation}/{childIndex}_beforeFrequency{particle.frequency}_afterFrequency{frequency}.txt",
                                     f"""
Hello Felix! I am a program.
                                     this is how many times I was originally sampled (how many parents gave birth to me): {particle.frequency}
                                     but then Darwin a.k.a.the value function stepped in and changed my frequency to: {frequency}
                                     this is how much Darwin a.k.a.the value function liked me: {-particle.distance}
here is my source code:
{particle.graph.prettyPrint(letters=True)}

and here is a trace of every command that gave rise to me:
                                     {trajectory}
""")
                        for ri,pr in enumerate(particle.graph.objects()):
                            # 2d
                            #pr.export(f"experimentOutputs/SMC/{numberOfParticles}/generation{generation}/{childIndex}_beforeFrequency{particle.frequency}_afterFrequency{frequency}_canvas{ri}.png",
                            #          256)
                            pr.scad(f"experimentOutputs/SMC/{INSTRUMENT}/{numberOfParticles}/generation{generation}/{childIndex}_beforeFrequency{particle.frequency}_afterFrequency{frequency}_canvas{ri}.png")
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

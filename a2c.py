from ForwardSample import *
from API import *
from pointerNetwork import *
from programGraph import *

import torch.nn.functional as F
import numpy as np


class A2C:
    def __init__(self, model, outerBatch=2, innerBatch=16):
        self.model = model
        self.outerBatch = outerBatch
        self.innerBatch = innerBatch

    def train(self, checkpoint, getSpec, R):
        fs = ForwardSample(self.model)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, eps=1e-3, amsgrad=True)

        value_losses = []
        policy_losses = []
        lastUpdate = 0
        updateFrequency = 100
        
        while True:
            specs = [getSpec() for _ in range(self.outerBatch) ]

            t0 = time.time()
            specEncodings = self.model.specEncoder(np.array([s.execute() for s in specs ]))
            objectEncodings = ScopeEncoding(self.model)
            fs.maximumLength = 1 + max(len(spec) for spec in specs)
            trajectories = fs.batchedRollout(specs, self.innerBatch,
                                             objectEncodings=objectEncodings,
                                             specEncodings=specEncodings)
            gs = [ [ProgramGraph(t) for t in ts ]
                   for ts in trajectories ]
            # batchSuccess = 0
            # oldSuccess = 0
            # for spec,graphs in zip(specs,gs):

            #     print("for the spec",spec)
            #     for g in graphs:
            #         print(g.prettyPrint())
            #         print(R(spec,g))
            #         batchSuccess += int(R(spec,g))
            #         print()
            #     print()
            #     print("Without batching...")
            #     for _ in range(self.innerBatch):
            #         g = fs.rollout(spec)
            #         if g is None: continue
                    
            #         print(g.prettyPrint())
            #         print(R(spec,g))
            #         oldSuccess += int(R(spec,g))
            #         print()
            #     print()
            #     print()
            # print("COMPARE\t",batchSuccess,oldSuccess)
                
            successes = [ [1.*int(R(spec, g)) for g in _g]
                          for spec,_g in zip(specs, gs) ]

            

            # Build training targets for value
            # Jointly build the vectorized input for the distance head
            valueTrainingTargets = []
            distanceInput = []
            for si,(spec,ts) in enumerate(zip(specs, trajectories)):
                for ti,trajectory in enumerate(ts):
                    succeeded = successes[si][ti]
                    for t in range(len(trajectory) + 1):
                        g = ProgramGraph(trajectory[:t])
                        objects = g.objects(oneParent=self.model.oneParent)
                        oe = objectEncodings.encoding(spec, objects)
                        valueTrainingTargets.append(float(int(succeeded)))
                        distanceInput.append((specEncodings[si], oe))

            distancePredictions = self.model.batchedDistance([oe for se,oe in distanceInput],
                                                             [se for se,oe in distanceInput])
            distanceTargets = self.model.tensor(valueTrainingTargets)
            value_loss = binary_cross_entropy(-distancePredictions, distanceTargets, average=False)

            # REINFORCE objective
            reinforcedLikelihoods = []
            for si,(spec,ts) in enumerate(zip(specs, trajectories)):
                usedTrajectories = set()
                successfulTrajectories = sum(successes[si][ti] > 0.
                                             for ti in range(len(ts)) )
                for ti,trajectory in enumerate(ts):
                    if successes[si][ti] > 0.:
                        if tuple(trajectory) in usedTrajectories: continue
                        
                        usedTrajectories.add(tuple(trajectory))
                        frequency = sum(tp == trajectory for tp in ts)
                        
                        ll = self.model.traceLogLikelihood(spec, trajectory,
                                                           scopeEncoding=objectEncodings,
                                                           specEncoding=specEncodings[si])[0]
                        reinforcedLikelihoods.append(ll*(frequency/successfulTrajectories))
                if successfulTrajectories == 0:
                    # mix imitation with REINFORCE
                    ll = self.model.traceLogLikelihood(spec, spec.toTrace(),
                                                       scopeEncoding=objectEncodings,
                                                       specEncoding=specEncodings[si])[0]
                    reinforcedLikelihoods.append(ll)
            if reinforcedLikelihoods:
                policy_loss = -sum(reinforcedLikelihoods)/len(reinforcedLikelihoods)
            else:
                policy_loss = None

            self.model.zero_grad()
            if policy_loss is None:
                value_loss.backward()
            else:
                (value_loss + policy_loss).backward()
                policy_losses.append(policy_loss.cpu().data.item())
            optimizer.step()
            value_losses.append(value_loss.cpu().data.item())
            
            lastUpdate += 1
            
            if lastUpdate%updateFrequency == 1:
                print(f"Average value loss: {sum(value_losses)/len(value_losses)}")
                if policy_losses:
                    print(f"Average policy loss: {sum(policy_losses)/len(policy_losses)}")
                
                policy_losses, value_losses = [], []
                torch.save(self.model, checkpoint)

                print("Live update of model predictions!")
                k = 0
                for si,spec in enumerate(specs):
                    print(spec)
                    print()

                    for ti,trajectory in enumerate(trajectories[si]):
                        print()
                        print(f"TRAJECTORY #{ti}: Success? {valueTrainingTargets[k]}")
                        for t in range(len(trajectory) + 1):
                            
                            if t == 0:
                                print(f"Prior to any actions, predicted distance is {distancePredictions[k]}")
                            else:
                                print(f"After taking action {trajectory[t - 1]}\t\t{distancePredictions[k]}\tadvantage {math.exp(-distancePredictions[k]) - math.exp(-distancePredictions[k - 1])}")

                            k += 1

                    print()

                assert k == len(valueTrainingTargets)
                    
                    

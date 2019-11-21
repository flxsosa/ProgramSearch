from ForwardSample import *
from API import *
from pointerNetwork import *
from programGraph import *

import torch.nn.functional as F
import numpy as np


class A2CAbstract:
    """
    We have 4 losses.
    training the policy is easy. 
    
    """


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
            fs.maximumLength = 1 + max(len(spec.toTrace()) for spec in specs)
            trajectories = fs.batchedRollout(specs, self.innerBatch,
                                             objectEncodings=objectEncodings,
                                             specEncodings=specEncodings)
            gs = [ [ProgramGraph(t) for t in ts ]
                   for ts in trajectories ]

                
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
                    if self.model.abstract:
                        ll = self.model.traceLogLikelihood(spec, spec.abstract().toTrace(),
                                   scopeEncoding=objectEncodings,
                                   specEncoding=specEncodings[si])[0]
                    else:
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
     




def compute_value_loss():


    distancePredictions = self.model.batchedDistance([oe for se,oe in distanceInput],
                                                     [se for se,oe in distanceInput])
    distanceTargets = self.model.tensor(valueTrainingTargets)
    value_loss = binary_cross_entropy(-distancePredictions, distanceTargets, average=False) 
                  



def getNegativeExample(spec):
    pass
    #should it output a trace or not? does it matter?
    #currently requires a trace



def trainAbstractContrastive(m, getProgram, trainTime=None, checkpoint=None, loss_mode='cross_entropy', example_mode='posNegTraces'):
    #assert mode=='cross_entropy'
    print("cuda?", m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    iteration = 0

    B = 16

    while trainTime is None or time.time() - startTime < trainTime:
        sys.stdout.flush()

        #possibly refactor
        ss = [getProgram() for _ in range(B)]
        ss = [(spec, s.abstract().toTrace(), getNegativeExample(spec) ) for spec in ss] 

        ls = m.gradientStepContrastiveBatched(optimizer, ss, loss_mode=loss_mode, example_mode=example_mode)


        #TODO deal with printing the stuff I want
        for l in ls:
            totalLosses.append(sum(l))
            movedLosses.append(sum(l)/len(l))
        iteration += 1
        if iteration%reportingFrequency == 1:
            print(f"\n\nAfter {iteration*B} training examples...\n\tTrace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}\n{iteration*B/(time.time() - startTime)} examples/sec\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            torch.save(m, checkpoint)


def gradientStepContrastiveBatched(self, optimizer, specsPosNegTraces, loss_mode='cross_entropy', example_mode='posNegTraces'):
    """specsPosNegTraces is an object of form: #TODO"""

    if example_mode == 'posNegSpecs':
        return gradientStepContrastiveBatchedPosNegSpecs(optimizer, 
                                                        specsPosNegTraces, 
                                                        loss_mode=loss_mode, 
                                                        example_mode=example_mode)

    # in this version, we have positive and negative trace examples
    # compute policy loss
    specsAndPositiveTraces = [(spec, posTrace) for spec, posTrace, _ in specsPosNegTraces] #this changes per mode
    policy_loss, policy_losses_list, specEncodings = self.computePolicyLossTraceBatched(optimizer, specsAndPositiveTraces)

    # register negative examples
    scopeEncoding.registerObjects([(o,spec) #this changes per mode
                               for spec, _, negTrace in specsPosNeg
                               for o in negTrace ]) 
    distanceInput = []
    valueTrainingTargets = []
    for b, (spec, posTrace, negTrace) in enumerate(specsPosNegTraces):

        #how does this work? can i just do:
        for t in range(len(posTrace)+1):
            traceObjs = posTrace[:t]
            oe_pos = self.getOE(spec, traceObjs, scopeEncoding)
            distanceInput.append(specEncodings[b], oe_pos)
            valueTrainingTargets.append(1.0)

        for t in range(len(negTrace)+1):
            traceObjs = negTrace[:t]
            oe_neg = self.getOE(spec, traceObjs, scopeEncoding)
            distanceInput.append(specEncodings[b], oe_neg)
            valueTrainingTargets.append(0.0)

    #compute value loss
    distancePredictions = self.batchedDistance([oe for se,oe in distanceInput],
                                                     [se for se,oe in distanceInput])
    distanceTargets = self.tensor(valueTrainingTargets) #TODO

    if loss_mode=='cross_entropy':
        value_loss = binary_cross_entropy(-distancePredictions, distanceTargets, average=False) #does this mean it's not a sum??
    elif loss_mode == 'triplet':
        #NB: will only work if trace has same number of objects cannonicallized the same way or something ...
        # BIG HACK, they happen to line up bc of previous code

        positivePredictions = distancePredictions[distanceTargets == 1.]
        negativePredictions = distancePredictions[distanceTargets == 0.]

        value_loss = triplet_loss(positivePredictions, negativePredictions)
    else: assert False, f"mode {mode} not implemented"


    (value_loss + policy_loss).backward()
    optimizer.step()
    return value_loss.cpu().data.item(), policy_loss.cpu().data.item()


def gradientStepContrastiveBatchedPosNegSpecs(self, optimizer, PosNegSpecsAndTraces, loss_mode='cross_entropy'):
    """specsPosNeg is an object of form: #TODO"""
    # in this version, we have positive and negative Specs, and a single set of Traces

    # compute policy loss
    PosSpecsAndTraces = [(spec, trace) for posSpec, _, posTrace, in PosNegSpecsAndTraces] #this changes per mode
    policy_loss, policy_losses_list, posSpecEncodings = self.computePolicyLossTraceBatched(optimizer, PosSpecsAndTraces)


    #compute negative spec encodings:
    negSpecRenderings = np.array([s.execute()
                                   for _,s , _ in PosNegSpecsAndTraces ])
    negSpecEncodings = self.specEncoder(negSpecRenderings)

    # register negative specexamples
    scopeEncoding.registerObjects([(o,spec) #this changes per mode
                               for _, negSpec, trace in PosNegSpecsAndTraces
                               for o in trace ])
    distanceInput = []
    valueTrainingTargets = []
    for b, (posSpec, negSpec, trace) in enumerate(specsPosNeg):
        #how does this work? can i just do:
        for t in range(len(trace)+1):
            traceObjs = trace[:t]
            oe_pos = self.getOE(posSpec, traceObjs, scopeEncoding)
            distanceInput.append(posSpecEncodings[b], oe_pos)
            valueTrainingTargets.append(1.)

            oe_neg = self.getOE(negSpec, traceObjs, scopeEncoding)
            distanceInput.append(negSpecEncodings[b], oe_neg)
            valueTrainingTargets.append(0.)

    #compute value loss
    distancePredictions = self.batchedDistance([oe for se,oe in distanceInput],
                                                     [se for se,oe in distanceInput])
    distanceTargets = self.tensor(valueTrainingTargets)

    if loss_mode=='cross_entropy':
        value_loss = binary_cross_entropy(-distancePredictions, distanceTargets, average=False) #does this mean it's not a sum??
    elif loss_mode == 'triplet':
        # BIG HACK, they happen to line up bc of previous code
        positivePredictions = distancePredictions[distanceTargets == 1.]
        negativePredictions = distancePredictions[distanceTargets == 0.]

        value_loss = triplet_loss(positivePredictions, negativePredictions)
    else: assert False, f"mode {mode} not implemented"
    #todo
    (value_loss + policy_loss).backward()
    optimizer.step()
    return value_loss.cpu().data.item(), policy_loss.cpu().data.item()





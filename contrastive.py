from API import *
from pointerNetwork import *
from programGraph import *

import torch.nn.functional as F
import numpy as np

from CAD import changeTopLevel, Union,Ab_Circle, Ab_Rectangle, Difference,rewrite
import sys

def mutate(prog, p=0.2): #TODO modify for concrete programs
    """
    nb type(prog) and prog.type are different
    type(prog) gives the value of the root node
    prog.type gives the type sig of the root node
    """
    if random.random() < p:
        return changeTopLevel(prog)
    else:
        return type(prog)(*map(mutate, prog.children()))

def getNegativeExample(prog, abstract=True):
    """
    randomly permute an abstract program to get a different one
    requres that input prog is abstract
    """
    if abstract:
        prog = prog.abstract()
    else:
        assert False
    negProg = mutate(prog)
    return negProg


def getRewrite(prog):
    prog = prog.abstract()
    posProg = rewrite(prog)
    return posProg

def trainAbstractContrastive(m,
                             getProgram,
                             trainTime=None,
                             checkpoint=None,
                             loss_mode='cross_entropy',
                             vector_loss_type=None,
                             example_mode='posNegTraces',
                             train_abstraction=True, alpha=0.2):
    #assert mode=='cross_entropy'
    print("cuda?", m.use_cuda)
    assert checkpoint is not None, "must provide a checkpoint path to export to"
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001, eps=1e-3, amsgrad=True)
    
    startTime = time.time()
    reportingFrequency = 100
    totalLosses = []
    movedLosses = []
    value_losses = []
    vector_losses = []
    iteration = 0
    B = 16


    while trainTime is None or time.time() - startTime < trainTime:
        sys.stdout.flush()

        #possibly refactor
        ss = [getProgram() for _ in range(B)]
        if train_abstraction:
            ss = [(spec, spec.abstract().toTrace(), getNegativeExample(spec).toTrace() ) for spec in ss] 
        else:
            ss = [(spec, spec.toTrace(), getNegativeExample(spec, abstract=False).toTrace() ) for spec in ss] #TODO

        policy_loss, value_loss, vector_loss, policy_losses = m.gradientStepContrastiveBatched(optimizer, ss, 
                                                                                loss_mode=loss_mode, 
                                                                                example_mode=example_mode,
                                                                                vector_loss_type=vector_loss_type,
                                                                                iteration=iteration, alpha=alpha)

        value_losses.append(value_loss)
        vector_losses.append(vector_loss)
        for l in policy_losses:
            totalLosses.append(sum(l))
            movedLosses.append(sum(l)/len(l))
        iteration += 1
        if iteration%reportingFrequency == 1:
            print(f"\n\nAfter {iteration*B} training examples...")
            print(f"\tPolicy: Trace loss {sum(totalLosses)/len(totalLosses)}\t\tMove loss {sum(movedLosses)/len(movedLosses)}")
            print(f"\tValue: Average loss: {sum(value_losses)/len(value_losses)}")
            print(f"\tVector: Average loss: {sum(vector_losses)/len(vector_losses)}")
            print(f"{iteration*B/(time.time() - startTime)} examples/sec\n{iteration/(time.time() - startTime)} grad steps/sec")
            totalLosses = []
            movedLosses = []
            value_losses = []
            torch.save(m, checkpoint)




if __name__=='__main__':
    p = Difference(Ab_Circle(), Union(Ab_Rectangle(),Ab_Circle()))

    #print(mutate(p))
    x = rewrite(p)
    print(x)
    print(rewrite(x) == p)
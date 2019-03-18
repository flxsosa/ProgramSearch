import pregex as pre
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch import optim
import string
#from collections import Counter

print_every=25
names = ["Luke Hewitt", "Max Nye", "Kevin Ellis", "Josh Rule", "Josh Tenenbaum"]

for mode in ["unigram", "bigram", "trigram"]:
    print("\n------" + mode + "------")
    a = string.ascii_lowercase + string.ascii_uppercase + " "

    if mode=="unigram":
        keys = [""]
    elif mode=="bigram":
        keys = [""] + list(a)
    elif mode=="trigram":
        keys = [""] + list(a) + [a1 + a2 for a1 in a for a2 in a]
    
    t_l = Parameter(torch.zeros(len(keys), len(a)))
    t_p = Parameter(torch.zeros(len(keys)))

    optimiser = optim.Adam([t_l, t_p], lr=0.1)

    for i in range(501):
        optimiser.zero_grad()
        ps_tens = F.softmax(t_l, dim=1)
        p_tens = F.sigmoid(t_p)
        ps = {k:v for k,v in zip(keys, ps_tens)}
        p = {k:v for k,v in zip(keys, p_tens)}

        l = pre.CharacterClass(a, name="alphanumeric", ps=ps, normalised=True)
        word = pre.Plus(l, p=p)
        r = pre.create("w w", lookup={"w":word})
        score = sum(r.match(name) for name in names)
        (-score).backward(retain_graph=True)
        optimiser.step()

        if i%print_every == 0:
            print("Iteration %3d | Score %7.2f | %s" %(i, score.item(), r.sample()))


    #c = Counter(s)
    #for i,x in enumerate(a):
    #    assert((ps[i] - (c[x] / len(c))) < 0.01)
    #assert((p-1/(len(s)+1)).abs() < 0.01)

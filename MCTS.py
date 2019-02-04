from programGraph import *
import time


class MCTS():
    def __init__(self, model, _=None, 
                 beamSize=10, discountFactor=0.9, cb=1, ca=100, rolloutDepth=None, reward=None, defaultTimeout=None):
        assert reward is not None, "must specify reward: spec X graph -> real"
        self.defaultTimeout = defaultTimeout
        self.discountFactor = discountFactor
        self.reward = reward        
        self.ca = ca
        self.cb = cb
        self.beamSize = beamSize
        self.model = model
        self.rolloutDepth = rolloutDepth

        self.beamTime = 0.
        self.distanceTime = 0.
        self.rollingTime = 0.

            

    def infer(self, spec, timeout=None):
        if timeout is None: timeout = self.defaultTimeout
        with torch.no_grad(): return self._infer(spec, timeout)
    
    def _infer(self, spec, timeout):
        startTime = time.time()

        class Node:
            def __init__(self, graph, predictedDistance):
                self.graph = graph
                self.predictedDistance = predictedDistance
                self.visits = 0
                self.edges = []

        class Edge:
            def __init__(self, parent, child, logLikelihood):
                self.logLikelihood = logLikelihood
                self.parent = parent
                self.child = child
                self.traversals = 0
                self.totalReward = 0
                self.totalValue = 0

        bestReward = None
        bestProgram = None
        def recordProgram(g):
            nonlocal bestReward
            nonlocal bestProgram
            r = self.reward(spec, g)
            if bestReward is None or r > bestReward:
                bestReward = r
                bestProgram = g

        specEncoding = self.model.specEncoder(spec)
        objectEncodings = ScopeEncoding(self.model, spec)

        # Maps from a graph to its distance
        _distance = {}
        def distance(g):
            if g in _distance: return _distance[g]
            se = objectEncodings.encoding(list(g.objects()))
            t0 = time.time()
            d = self.model.distance(se, specEncoding).data.item()
            self.distanceTime += time.time() - t0
            _distance[g] = d
            return d

        def expand(n):
            t0 = time.time()
            bm = self.model.beamNextLine(specEncoding, n.graph, objectEncodings, self.beamSize)
            self.beamTime += time.time() - t0
            
            for o, ll in bm:
                if o is None or o in n.graph.nodes: continue
                newGraph = n.graph.extend(o)
                if newGraph in graph2node:
                    child = graph2node[newGraph]
                else:
                    recordProgram(newGraph)
                    child = Node(newGraph, distance(newGraph))
                e = Edge(n, child, ll)
                n.edges.append(e)

        def rollout(g):
            t0 = time.time()
            depth = 0
            while True:
                samples = self.model.repeatedlySample(specEncoding, g, objectEncodings, 1)
                assert len(samples) <= 1
                depth += 1
                if len(samples) == 0 or samples[0] is None: break
                g = g.extend(samples[0])
                if self.rolloutDepth is not None and depth >= self.rolloutDepth: break

            self.rollingTime += time.time() - t0
            recordProgram(g)
            
            return g

        def uct(e):
            if e.traversals == 0: return float('inf')

            # Exploit: Weighted average of rewards and distance
            wv = 0.
            confidence = (1. - wv)*e.totalReward/e.traversals + wv*e.totalValue/e.traversals
            # Explore: Prefer paths that are less visited
            confidence += self.cb*(math.log(e.parent.visits)/e.traversals)**0.5
            # Policy: Prefer paths the neural net likes
            confidence += self.ca*math.exp(e.logLikelihood)/(e.traversals + 1)
            return confidence

        rootNode = Node(ProgramGraph([]), distance(ProgramGraph([])))
        graph2node = {ProgramGraph([]): rootNode}

        while time.time() - startTime < timeout:
            n = rootNode
            trajectory = [] # list of traversed edges
            while n.visits > 0 and len(n.edges) > 0:
                e = max(n.edges, key=uct)
                trajectory.append(e)
                n = e.child

            d = distance(n.graph)
            r = self.reward(spec, rollout(n.graph))
            # back up the reward
            for e in trajectory:
                e.totalReward += r
                e.totalValue += self.discountFactor**d
                e.traversals += 1
                e.parent.visits += 1

            if n.visits == 0:
                expand(n)
                n.visits = 1
            else:
                n.visits += 1                

        return bestProgram
                         
        

        

from API import *
from programGraph import *
from pointerNetwork import *
import time


class MCTS(Solver):
    """
    AlphaZero-style Monte Carlo tree search
    Currently ignores learned distance / value, but is biased by learned policy
    """
    def __init__(self, model, _=None, reward=None,
                 c_puct=5, rolloutDepth=None):
        """
        c_puct: Trades off exploration and exploitation. Larger values favor exploration, guided by policy.
        reward: function from loss to reward.
        """
        assert reward is not None, "reward must be specified. This function converts loss into reward."
        self.reward = reward
        self.c_puct = c_puct
        self.model = model
        self.rolloutDepth = rolloutDepth

        self.beamTime = 0.
        self.rollingTime = 0.

    def __str__(self):
        return f"MCTS(puct={self.c_puct})"
            
    def _infer(self, spec, loss, timeout):
        startTime = time.time()
        owner = self

        class Node:
            def __init__(self, graph):
                self.graph = graph
                self.visits = 0
                self.edges = []
                self.generator = owner.model.bestFirstEnumeration(specEncoding, graph, objectEncodings)

        class Edge:
            def __init__(self, parent, child, logLikelihood):
                self.logLikelihood = logLikelihood
                self.parent = parent
                self.child = child
                self.traversals = 0
                self.totalReward = 0

        specEncoding = self.model.specEncoder(spec)
        objectEncodings = ScopeEncoding(self.model, spec)


        def expand(n):
            """Adds a single child to a node"""
            if n.generator is None: return 
            t0 = time.time()
            try: o, ll = next(n.generator)
            except StopIteration:
                n.generator = None
                o, ll = None, None
            self.beamTime += time.time() - t0
            
            if o is None or o in n.graph.nodes: return 
            newGraph = n.graph.extend(o)
            if newGraph in graph2node:
                child = graph2node[newGraph]
            else:
                self._report(newGraph)
                child = Node(newGraph)
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
            self._report(g)
            
            return g

        def uct(e):
            # Exploit: rewards Q(s,a)
            if e.traversals == 0: q = 0.
            else: q = e.totalReward/e.traversals

            # Explore, biased by policy
            exploration_bonus = math.exp(e.logLikelihood) * (e.parent.visits**0.5) / (1. + e.traversals)

            # Trade-off of exploration and exploitation
            return q + self.c_puct*exploration_bonus
        
        rootNode = Node(ProgramGraph([]))
        graph2node = {ProgramGraph([]): rootNode}

        while time.time() - startTime < timeout:
            n = rootNode
            trajectory = [] # list of traversed edges

            while len(n.edges) > 0:
                e = max(n.edges, key=uct)
                trajectory.append(e)
                n = e.child

            r = self.reward(self.loss(rollout(n.graph)))

            # Expand nodes if their single visit-0 child was visited
            for e in trajectory:
                if e.child.visits == 0:
                    expand(e.parent)
            
            # back up the reward
            for e in trajectory:
                e.totalReward += r
                e.traversals += 1
                e.parent.visits += 1

            expand(n)
            n.visits += 1                

                         
        

        

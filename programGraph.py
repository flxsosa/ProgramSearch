from API import *
from utilities import *

class ProgramGraph:
    """A program graph is a state in the search space"""
    def __init__(self, nodes):
        self.nodes = nodes if isinstance(nodes, frozenset) else frozenset(nodes)

    @staticmethod
    def fromRoot(r):
        ns = set()
        def reachable(n):
            if n in ns: return
            ns.add(n)
            for c in n.children():
                reachable(c)
        reachable(r)
        return ProgramGraph(ns)

    def __len__(self): return len(self.nodes)

    def prettyPrint(self):
        index2node = []
        node2index = {}
        index2code = {}
        def getIndex(n):
            if n in node2index: return node2index[n]
            serialization = [ t if not isinstance(t, Program) else f"${getIndex(t)}"
                              for t in n.serialize() ]
            myIndex = len(index2node)
            index2node.append(n)
            index2code[myIndex] = "(" + " ".join(map(str, serialization)) + ")"
            node2index[n] = myIndex            
            return myIndex
        for n in self.nodes: getIndex(n)
        return "\n".join( f"${i} <- {index2code[i]}"
                          for i in range(len(index2node)))
                          
            
            

    def extend(self, newNode):
        return ProgramGraph(self.nodes | {newNode})

    def objects(self):
        return self.nodes

    def policyOracle(self, currentGraph):
        """Takes the current graph and returns moves that take you closer to the goal graph (self)"""
        missingNodes = self.nodes - currentGraph.nodes
        for n in missingNodes:
            if all( child in currentGraph.nodes for child in n.children() ):
                yield n

    def distanceOracle(self, targetGraph):
        return len(self.nodes^targetGraph.nodes)


from API import *
from utilities import *

class ProgramGraph:
    """A program graph is a state in the search space"""
    def __init__(self, nodes):
        self.nodes = nodes if isinstance(nodes, tuple) else tuple(nodes)

    @staticmethod
    def fromRoot(r, oneParent=False):
        if not oneParent:
            ns = set()
            def reachable(n):
                if n in ns: return
                ns.add(n)
                for c in n.children():
                    reachable(c)
            reachable(r)
            return ProgramGraph(ns)
        else:
            ns = []
            def visit(n):
                ns.append(n)
                for c in n.children(): visit(c)
            visit(r)
            return ProgramGraph(ns)
                    

    def __len__(self): return len(self.nodes)

    def prettyPrint(self):
        variableOfNode = [None for _ in self.nodes]
        nameOfNode = [None for _ in self.nodes] # pp of node

        lines = []

        def getIndex(p):
            for i, pp in enumerate(self.nodes):
                if p is pp: return i
            assert False                

        def pp(j):
            if variableOfNode[j] is not None: return variableOfNode[j]
            serialization = [t if not isinstance(t,Program) else pp(getIndex(t))
                             for t in self.nodes[j].serialize()]
            expression = f"({' '.join(map(str, serialization))})"
            variableOfNode[j] = f"${len(lines)}"
            lines.append(f"{variableOfNode[j]} <- {expression}")
            return variableOfNode[j]

        for j in range(len(self.nodes)):
            pp(j)
        return "\n".join(lines)
                          
    def extend(self, newNode):
        return ProgramGraph([newNode] + list(self.nodes))

    def objects(self, oneParent=False):
        return [o for o in self.nodes
                if not oneParent or not any( any( c is o for c in op.children() ) for op in self.nodes )]

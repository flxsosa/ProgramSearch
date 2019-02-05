import torch
import time

class Solver:
    def __init__(self):
        pass

    def _report(self, program):
        l = self.loss(program)
        if len(self.reportedSolutions) == 0 or self.reportedSolutions[-1].loss > l:
            self.reportedSolutions.append(SearchResult(program, l, time.time() - self.startTime))            
        
    def infer(self, spec, loss, timeout):
        """
        spec: specification of goal
        loss: function from (spec, program) to real
        timeout: maximum time to run solver, measured in seconds
        returns: list of `SearchResult`s
        Should take no longer than timeout seconds."""
        self.reportedSolutions = []
        self.startTime = time.time()
        self.loss = lambda p: loss(spec, p)

        with torch.no_grad():
            self._infer(spec, loss, timeout)

        self.loss = None # in case we need to serialize this object and loss is a lambda
        
        return self.reportedSolutions

    def _infer(self, spec, loss, timeout):
        assert False, "not implemented"

class SearchResult:
    def __init__(self, program, loss, time):
        self.program = program
        self.loss = loss
        self.time = time

class DSL:
    def __init__(self, operators, lexicon=None):
        """
        operators: a list of classes that inherit from Program
        lexicon: (optionally) a list of symbols in the serialization of programs built from those operators
        """
        self.lexicon = lexicon
        self.operators = operators

        self.tokenToOperator = {o.token: o
                                for o in operators}

    def parseLine(self, tokens):
        """
        Parses a serialized line of code into a Program object.
        Returns None if the DSL cannot parse the serialized code.
        """
        if len(tokens) == 0 or tokens[0] not in self.tokenToOperator: return None

        f = self.tokenToOperator[tokens[0]]
        if len(tokens) != len(f.argument_types) + 1: return None

        for token, argument_type in zip(tokens, f.argument_types):
            if not isinstance(token, argument_type): return None

        return f(*tokens[1:])

    
            


class Program:
    def execute(self, context):
        assert False, "not implemented"

    def children(self):
        assert False, "not implemented"




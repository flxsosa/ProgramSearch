import torch
import time

class Solver:
    def __init__(self, dsl):
        self.dsl = dsl
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


class ParseFailure(Exception):
    """Objects of type Program should throw this exception in their constructor if their arguments are bad"""

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

        for token, argument_type in zip(tokens[1:], f.argument_types):
            if not isinstance(token, argument_type): return None
        try:
            return f(*tokens[1:])
        except ParseFailure: return None

class Program:

    # TODO: implement type property
    @abstractproprty
    type = None

    def execute(self, context):
        assert False, "not implemented"

    def children(self):
        assert False, "not implemented"


class Type():
    pass

class BaseType(Type):
    def __init__(self, thing):
        self.constructor = thing

class arrow(Type):
    def __init__(self, *args):
        assert len(args) > 1
        for a in args:
            assert isinstance(a, Type)
        self.out = args[-1]
        self.arguments = args[:-1]

    def pretty_print(self):
        print("Arguments: {}".format(self.arguments))
        print("Output: {}".format(self.out))

class integer(Type):

    def __init__(self,lower,upper):
        assert type(lower) is int
        assert type(upper) is int
        self.upper = upper
        self.lower = lower


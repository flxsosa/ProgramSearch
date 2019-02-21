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

    def __str__(self):
        return "DSL({%s})"%(", ".join( f"{o.__name__} : {str(o.type)}"
                                       for o in self.operators ))

    def parseLine(self, tokens):
        """
        Parses a serialized line of code into a Program object.
        Returns None if the DSL cannot parse the serialized code.
        """
        if len(tokens) == 0 or tokens[0] not in self.tokenToOperator: return None

        f = self.tokenToOperator[tokens[0]]
        ft = f.type

        if ft.isArrow: # Expects arguments
            # Make sure we have the right number of arguments
            tokens = tokens[1:]
            if len(tokens) != len(ft.arguments): return None
            # Make sure that each token is an instance of the correct type
            for token, argument_type in zip(tokens, ft.arguments):
                if not argument_type.instance(token): return None
            # Type checking succeeded - try building the object
            try:
                return f(*tokens)
            except ParseFailure: return None
        else: # Does not expect any arguments - just call the constructor with no arguments
            if len(tokens) > 1: return None # got arguments when we were not expecting any
            return f()

class Program:

    # TODO: implement type property

    def execute(self, context):
        assert False, "not implemented"

    def children(self):
        assert False, "not implemented"


class Type():
    @property
    def isArrow(self): return False

    @property
    def isInteger(self): return False

    @property
    def isBase(self): return False

    def returnType(self):
        """What this type indicates the expression should return. For arrows this is the right-hand side. Otherwise it is just the type."""
        return self

class BaseType(Type):
    def __init__(self, thing):
        self.constructor = thing

    def __str__(self):
        return self.constructor.__name__

    @property
    def isBase(self): return False

    def instance(self, x):
        return isinstance(x, self.constructor)
    
class arrow(Type):
    def __init__(self, *args):
        assert len(args) > 1
        for a in args:
            assert isinstance(a, Type)
        self.out = args[-1]
        self.arguments = args[:-1]

    def __str__(self):
        return " -> ".join( str(t) for t in list(self.arguments) + [self.out] )

    @property
    def isArrow(self): return True

    def instance(self, x):
        assert False, "Cannot check whether a object is an instance of a arrow type"

    def returnType(self): return self.out

class integer(Type):
    def __init__(self,lower,upper):
        assert type(lower) is int
        assert type(upper) is int
        self.upper = upper
        self.lower = lower

    def __str__(self):
        return f"int({self.lower}, {self.upper})"
    
    @property
    def isInteger(self): return True

    def instance(self, x):
        return isinstance(x, int) and x >= self.lower and x <= self.upper

    

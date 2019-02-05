class Solver:
    def __init__(self):
        pass

    def infer(self, spec, loss, timeout):
        """
        spec: specification of goal
        timeout: maximum time to run solver, measured in seconds
        returns: list of (program, loss, time)
        Should take no longer than timeout seconds."""
        assert False, "not implemented"

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
        if len(tokens) == 0: return None

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




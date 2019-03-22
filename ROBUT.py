import pickle
import numpy as np

import random
import string
from string import printable
import re
import pregex as pre
from ROB import R, _INDEX, _DELIMITER

N_IO = 5

class RobState:
    @staticmethod
    def new(inputs, outputs):
        assert len(inputs) == len(outputs)
        committed = ["" for _ in range(N_IO)]
        scratch = [x for x in inputs]
        past_buttons = []
        return RobState(inputs, scratch, committed, outputs, past_buttons)

    def __init__(self, inputs, scratch, committed, outputs, past_buttons):

        self.inputs    = [x for x in inputs]
        self.scratch   = [x for x in scratch]
        self.committed = [x for x in committed]
        self.outputs   = [x for x in outputs]
        self.past_buttons = [x for x in past_buttons]

    def __repr__(self):
        return str((self.inputs, self.scratch, self.committed, self.outputs, self.past_buttons))

    def __str__(self):
        return self.__repr__()

    def to_np(self):
        pass

# ===================== BUTTONS ======================

class Button:

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.__repr__()

class Commit(Button):
    @staticmethod
    def generate_buttons():
        return Commit()

    def __init__(self):
        self.name = f"Commit"

    def __call__(self, pstate):
        scratch_new = pstate.inputs
        committed_new = [x[0]+x[1] for x in zip(pstate.committed, pstate.scratch)]
        return RobState(pstate.inputs,
                        scratch_new,
                        committed_new,
                        pstate.outputs,
                        pstate.past_buttons + [self])


class ToCase(Button):

    @staticmethod
    def generate_buttons():
        ss = ["Proper", "AllCaps", "Lower"]
        ret = [ToCase(s) for s in ss]
        return ret

    def __init__(self, s):
        self.name = f"ToCase({s})"

        if s == "Proper":
            self.f = lambda x: x.title()
        if s == "AllCaps":
            self.f = lambda x: x.upper()
        if s == "Lower":
            self.f = lambda x: x.lower()

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class Replace1(Button):
    @staticmethod
    def generate_buttons():
        return [Replace1(d1) for d1 in _DELIMITER]

    def __init__(self, d1):
        self.name = f"Replace1({d1})"
        self.d1 = d1

    def __call__(self, pstate):
        ret =  RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])
        ret.replace1 = self.d1
        return ret

class Replace2(Button):
    @staticmethod
    def generate_buttons():
        return [Replace2(d2) for d2 in _DELIMITER]

    def __init__(self, d2):
        self.name = f"Replace2({d2})"
        self.d2 = d2

    def __call__(self, pstate):
        d1 = pstate.replace1
        scratch_new = [x.replace(d1, self.d2) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])


class SubStr1(Button):

    @staticmethod
    def generate_buttons():
        k1s = range(-100, 101)
        ret = [SubStr1(k1) for k1 in k1s]
        return ret

    def __init__(self, k1):
        self.name = f"SubStr1({k1})"
        self.k1 = k1

    def __call__(self, pstate):
        ret =  RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])
        ret.substrk1 = self.k1
        return ret

class SubStr2(Button):

    @staticmethod
    def generate_buttons():
        k2s = range(-100, 101)
        ret = [SubStr2(k2) for k2 in k2s]
        return ret

    def __init__(self, k2):
        self.name = f"SubStr2({k2})"
        self.k2 = k2

    def __call__(self, pstate):
        k1 = pstate.substrk1
        scratch_new = [x[k1:self.k2] for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetToken(Button):

    @staticmethod
    def generate_buttons():
        return [GetToken(name, i) \
                for name in R.possible_types.keys() for i in _INDEX] 

    def __init__(self, rname, i):
        self.name = f"GetToken({rname}, {i})"
        self.rname = rname
        self.t = R.possible_types[rname]
        self.i = i
        self.f = lambda x: re.findall(self.t[0], x)[i]

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetUpTo(Button):

    @staticmethod
    def generate_buttons():
        return [GetUpTo(name) \
            for name in R.possible_r.keys()] 

    def __init__(self, rname):
        self.name = f"GetUpTo({rname})"
        self.rname = rname
        self.r = R.possible_r[rname]
        self.f = lambda string: string[:[m.end() \
                for m in re.finditer(self.r[0], string)][0]]

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFrom(Button):

    @staticmethod
    def generate_buttons():
        return [GetFrom(name) \
            for name in R.possible_r.keys()] 

    def __init__(self, rname):
        self.name = f"GetFrom({rname})"
        self.rname = rname
        self.r = R.possible_r[rname] 
        self.f = lambda string: string[[m.end() \
                for m in re.finditer(self.r[0], string)][-1]:]

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFirst(Button):

    @staticmethod
    def generate_buttons():
        return [GetFirst(name, i) \
        for name in R.possible_types.keys() for i in _INDEX] 

    def __init__(self, rname, i):
        self.name = f"GetFirst({rname}, {i})"
        self.rname = rname
        self.t = R.possible_types[rname]
        self.i = i
        self.f = lambda string: ''.join(re.findall(self.t[0], string)[:self.i])

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetAll(Button):

    @staticmethod
    def generate_buttons():
        return [GetAll(name) \
            for name in R.possible_types.keys()] 

    def __init__(self, rname):
        self.name = f"GetAll({rname})"
        self.rname = rname
        self.r = R.possible_types[rname]
        self.f = lambda string: ''.join(re.findall(self.r[0], string))

    def __call__(self, pstate):
        scratch_new = [self.f(x) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])


ALL_BUTTS = ToCase.generate_buttons() +\
            Replace1.generate_buttons() +\
            Replace2.generate_buttons() +\
            SubStr1.generate_buttons() +\
            SubStr2.generate_buttons() +\
            GetToken.generate_buttons() +\
            GetUpTo.generate_buttons() +\
            GetFrom.generate_buttons() +\
            GetFirst.generate_buttons() +\
            GetAll.generate_buttons()


# ===================== UTILS ======================
def apply_fs(pstate, funcs):
    if funcs == []:
        return pstate
    else:
        last_state = apply_fs(pstate, funcs[:-1])
        return funcs[-1](last_state)

def test1():
    pstate = RobState.new(["12A", "2A4", "A45", "4&6", "&67"],
                          ["12a", "2a4", "a45", "4[6", "[67"])
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    print (pstate)
    fs = [
            ToCase("Lower"),
            Replace1("&"),
            Replace2("["),
            SubStr1(1),
            SubStr2(2),
            Commit(),
            ]

    print (apply_fs(pstate, fs))

def test2():
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    pstate = RobState.new(["Mr.Pu", "Mr.Poo"],
                          ["Pu", "Poo"])
    fs = [
            GetToken("Word", 1),
            Commit(),
            ]
    print (apply_fs(pstate, fs))

    gs = [
            GetUpTo("."),
            Commit(),
            ]
    print (apply_fs(pstate, gs))

    hs = [
            GetFrom("."),
            Commit(),
            ]
    print (apply_fs(pstate, hs))

    ts = [
            GetFirst("Word", 2),
            Commit(),
            ]
    print (apply_fs(pstate, ts))

    vs = [
            GetAll("Word"),
            Commit(),
            ]
    print (apply_fs(pstate, vs))

if __name__ == '__main__':
    test2()


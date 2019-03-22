import pickle
import numpy as np

import random
import string
from string import printable
import re
import pregex as pre
from ROB import R, _INDEX, _DELIMITER, _CHARACTER

N_IO = 5

class RobState:
    @staticmethod
    def new(inputs, outputs):
        assert len(inputs) == len(outputs)
        committed = ["" for _ in range(inputs)]
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
        # get the k1 from the previous button
        k1 = pstate.past_buttons[-1].k1
        scratch_new = [x[k1:self.k2] for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetToken1(Button):

    @staticmethod
    def generate_buttons():
        return [GetToken1(name) for name in R.possible_types.keys()] 

    def __init__(self, rname):
        self.name = f"GetToken1({rname})"
        self.rname = rname
        self.t = R.possible_types[rname]

    def __call__(self, pstate):
        return  RobState(pstate.inputs,
                         pstate.scratch,
                         pstate.committed,
                         pstate.outputs,
                         pstate.past_buttons + [self])

class GetToken2(Button):

    @staticmethod
    def generate_buttons():
        return [GetToken2(i) for i in _INDEX] 

    def __init__(self, i):
        self.name = f"GetToken2({i})"
        self.i = i
        self.f = lambda x, t: re.findall(t[0], x)[i]

    def __call__(self, pstate):
        # get the type from GetToken1 button
        t = pstate.past_buttons[-1].t
        scratch_new = [self.f(x, t) for x in pstate.scratch]
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

class GetFirst1(Button):

    @staticmethod
    def generate_buttons():
        return [GetFirst1(name) for name in R.possible_types.keys()]

    def __init__(self, rname):
        self.name = f"GetFirst1({rname})"
        self.rname = rname
        self.t = R.possible_types[rname]

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetFirst2(Button):

    @staticmethod
    def generate_buttons():
        return [GetFirst2(i) for i in _INDEX] 

    def __init__(self, i):
        self.name = f"GetFirst2({i})"
        self.i = i
        self.f = lambda string, t: ''.join(re.findall(t[0], string)[:self.i])

    def __call__(self, pstate):
        t = pstate.past_buttons[-1].t
        scratch_new = [self.f(x, t) for x in pstate.scratch]
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

class GetSpan1(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan1(name) for name in R.possible_r.keys()]

    def __init__(self, rname):
        self.name = f"GetSpan1({rname})"
        self.rname = rname
        self.r1 = R.possible_r[rname]

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan2(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan2(i) for i in _INDEX] 

    def __init__(self, i1):
        self.name = f"GetSpan2({i1})"
        self.i1 = i1

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan3(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan3("Start"), GetSpan3("End")]

    def __init__(self, b1):
        self.name = f"GetSpan3({b1})"
        self.b1 = b1

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan4(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan4(name) for name in R.possible_r.keys()]

    def __init__(self, rname):
        self.name = f"GetSpan4({rname})"
        self.rname = rname
        self.r2 = R.possible_r[rname]

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan5(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan5(i) for i in _INDEX] 

    def __init__(self, i2):
        self.name = f"GetSpan5({i2})"
        self.i2 = i2

    def __call__(self, pstate):
        return RobState(pstate.inputs,
                        pstate.scratch,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class GetSpan6(Button):

    @staticmethod
    def generate_buttons():
        return [GetSpan6("Start"), GetSpan6("End")]

    def __init__(self, b2):
        self.name = f"GetSpan6({b2})"
        self.b2 = b2

        def f(input_str, r1, i1, b1, r2, i2, b2):
            """
            all complaints of this function please send to mnye@mit.edu
            evanthebouncy took no part in this :v
            """
            return input_str[[m.end() for m in re.finditer(r1[0], input_str)][i1] if b1 == "End" else [m.start() for m in re.finditer(r1[0], input_str)][i1] : [m.end() for m in re.finditer(r2[0], input_str)][i2] if b2 == "End" else [m.start() for m in re.finditer(r2[0], input_str)][i2]]

        self.f = f

    def __call__(self, pstate):
        r1 = pstate.past_buttons[-5].r1
        i1 = pstate.past_buttons[-4].i1
        b1 = pstate.past_buttons[-3].b1
        r2 = pstate.past_buttons[-2].r2
        i2 = pstate.past_buttons[-1].i2
        b2 = self.b2
        scratch_new = [self.f(x, r1, i1, b1, r2, i2, b2) for x in pstate.scratch]
        return RobState(pstate.inputs,
                        scratch_new,
                        pstate.committed,
                        pstate.outputs,
                        pstate.past_buttons + [self])

class Const(Button):

    @staticmethod
    def generate_buttons():
        return [Const(c) for c in _CHARACTER]

    def __init__(self, c):
        self.name = f"Const({c})"
        self.c = c

    def __call__(self, pstate):
        scratch_new = [self.c for x in pstate.scratch]
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
            GetToken1.generate_buttons() +\
            GetToken2.generate_buttons() +\
            GetUpTo.generate_buttons() +\
            GetFrom.generate_buttons() +\
            GetFirst1.generate_buttons() +\
            GetFirst2.generate_buttons() +\
            GetAll.generate_buttons() +\
            GetSpan1.generate_buttons() +\
            GetSpan2.generate_buttons() +\
            GetSpan3.generate_buttons() +\
            GetSpan4.generate_buttons() +\
            GetSpan5.generate_buttons() +\
            GetSpan6.generate_buttons() +\
            Const.generate_buttons()


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
            GetToken1("Word"),
            GetToken2(1),
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
            GetFirst1("Word"),
            GetFirst2(2),
            Commit(),
            ]
    print (apply_fs(pstate, ts))

    vs = [
            GetAll("Word"),
            Commit(),
            ]
    print (apply_fs(pstate, vs))

def test3():
    print (ALL_BUTTS)
    print (len(ALL_BUTTS))
    pstate = RobState.new(["(hello)123", "(mister)123"],
                          ["1234", "4"])
    fs = [
            GetSpan1("("),
            GetSpan2(0),
            GetSpan3("End"),
            GetSpan4(")"),
            GetSpan5(0),
            GetSpan6("Start"),
            ToCase("AllCaps"),
            Commit(),
            Const("R"),
            Commit(),
            Const("E"),
            Commit(),
            ]
    print (apply_fs(pstate, fs))

if __name__ == '__main__':
    test1()
    test2()
    test3()


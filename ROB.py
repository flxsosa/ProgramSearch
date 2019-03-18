import pickle
import numpy as np

from API import *

from pointerNetwork import *
from programGraph import *
import random
import string
from string import printable
import re
import pregex as pre

"""
IMPLEMENTATION OF THE LANGUAGE OF THE ROBUST FILL THING
ALSO INPUT / OUTPUT GENERATION
ALSO PROGRAM SAMPLING

THIS THING IS CRAZY AS FUCKKkKkkKKKkKk good luck y'all


Overall Design Choice : 

    all parts of the grammar is defiend as a class with class method "generate"
    which samples a random expression from that production node forward. perhaps
    the best view point is that a class is the non-terminal and has the capability of
    constructing sub-trees all the way down to the terminal level and the resulting
    tree can be evaluated
"""

_INDEX = list(range(-5, 6))
_CHARACTER = string.printable[:-4]
_DELIMITER = "& , . ? ! @ ( ) [ ] % { } / : ; $ # \" '".split(' ')
_BOUNDARY = ["Start", "End"]

N_EXPRS = 6

class P:
    
    """
    a program is a concat of multiple expressions
    """

    @staticmethod
    def generate():
        n_expr = random.randint(1, N_EXPRS)    
        return P([E.generate() for _ in range(n_expr)])

    def __init__(self, exprs):
        self.exprs = exprs
        self.constr = None
        for e in self.exprs:
            self.constr = add_constr(e.constr, self.constr)

    def str_execute(self, input_str):
        return "".join(e.str_execute(input_str) for e in self.exprs)

class E:
    """
    an expression :D
    F | N | N1(N2) | N(F) | ConstStr(c)
    """
    @staticmethod
    def generate():
        ee_choices = [
        lambda: F.generate(),
        lambda: N.generate(),
        lambda: Compose(N.generate(), N.generate()),
        lambda: Compose(N.generate(), F.generate()),
        lambda: ConstStr.generate(),
        ]
        return E(random.choice(ee_choices)())

    def __init__(self, ee):
        self.ee = ee
        self.constr = ee.constr

    def str_execute(self, input_str):
        return self.ee.str_execute(input_str)

class Compose:
    """
    chain 2 things together :3
    """

    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2
        self.constr = add_constr(f1.constr, f2.constr)

    def str_execute(self, input_str):
        return self.f1.str_execute(self.f2.str_execute(input_str))

class F:
    """
    SubString | GetSpan
    """
    @staticmethod
    def generate():
        ee_choices = [
        lambda: SubString.generate(),
        lambda: GetSpan.generate(),
        ]
        return E(random.choice(ee_choices)())

    def __init__(self, ee):
        self.ee = ee
        self.constr = ee.constr

    def str_execute(self, input_str):
        return self.ee.str_execute(input_str)

class SubString:
    """
    take substring from position k1 to k2
    """
    @staticmethod
    def generate():
        _POSITION_K = list(range(-100, 101))
        k1 = random.choice(_POSITION_K)
        k2 = random.choice(_POSITION_K)
        if k1 > k2:
            k1, k2 = k2, k1
        return SubString(k1, k2)

    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2
        self.constr = ({}, k2)

    def str_execute(self, input_str):
        return input_str[self.k1:self.k2]

class GetSpan:
    @staticmethod
    def generate():
        r1 = R.generate()
        r2 = R.generate()
        i1 = random.choice(_INDEX)
        i2 = random.choice(_INDEX)
        b1 = random.choice(_BOUNDARY)
        b2 = random.choice(_BOUNDARY)
        return GetSpan(r1, i1, b1, r2, i2, b2)

    def __init__(self, r1, i1, b1, r2, i2, b2):
        self.r1, self.i1, self.b1 = r1, i1, b1
        self.r2, self.i2, self.b2 = r2, i2, b2

        dic = {r1 : stepped_abs(i1), 
               r2 : stepped_abs(i2), 
              }
        self.constr = dic, 0

    def __str__(self):
        return str((self.r1, self.i1, self.b1, self.r2, self.i2, self.b2))

    def str_execute(self, input_str):
        """
        all complaints of this function please send to mnye@mit.edu
        evanthebouncy took no part in this :v
        """
        return input_str[[m.end() for m in re.finditer(self.r1[0], input_str)][self.i1] if self.b1 == "End" else [m.start() for m in re.finditer(self.r1[0], input_str)][self.i1] : [m.end() for m in re.finditer(self.r2[0], input_str)][self.i2] if self.b2 == "End" else [m.start() for m in re.finditer(self.r2[0], input_str)][self.i2]]

class N:
    @staticmethod
    def generate():
        possible_nesting = {
            1:2
                }

class GetToken:

    @staticmethod
    def generate():
        t = R.generate_type()
        i = random.choice(_INDEX)
        return GetToken(t, i)

    def __init__(self, t, i):
        self.t, self.i = t, i

        dic = {t : stepped_abs(i), 
              }
        self.constr = dic, 0

    def __str__(self):
        return str((self.t, self.i))

    def str_execute(self, input_str):
        return re.findall(self.t[0], input_str)[self.i]

class ToCase:
    @staticmethod
    def generate():
        candidates = [
                lambda x : x.title(),
                lambda x: x.upper(),
                lambda x: x.lower(),
                ]
        return ToCase(random.choice(
    def __init__(self, s):


class R:

    @staticmethod
    def generate_type():
        possible_types = {
            "Number" :   (r'\d+', pre.create('\\d+')),
            "Word" :     (r'\w+', pre.create('\\w+')),
            "Alphanum" : (r'\w', pre.create('\\w')),
            "PropCase" : (r'[A-Z][a-z]+', pre.create('\\u\\l+')),
            "AllCaps" :  (r'[A-Z]', pre.create('\\u')),
            "Lower" :    (r'[a-z]', pre.create('\\l')),
            "Digit" :    (r'\d', pre.create('\\d')),
            "Char" :     (r'.', pre.create('.')),
            }
        type_choice = random.choice(list(possible_types.keys()))
        return R(possible_types[type_choice])

    @staticmethod
    def generate_delim():
        possible_delims = {}
        for i in _DELIMITER:
            j = i
            if j in ['(', ')', '.']: 
                j = re.escape(j)
            possible_delims[i] = (re.escape(i), pre.create(j))
        delim_choice = random.choice(list(possible_delims.keys()))
        return R(possible_delims[delim_choice])


    @staticmethod
    def generate():
        if np.random.random() < 0.5:
            return R.generate_type()
        else:
            return R.generate_delim()

    def __init__(self, regex):
        self.ree, self.pre = regex

    def __getitem__(self, key):
        if key == 0 : 
            return self.ree
        if key == 1 :
            return self.pre
        assert 0, "you ve gone too far"

    def __str__(self):
        return str((self.ree, self.pre))

    def __repr__(self):
        return self.__str__()

    def str_execute(self, input_str):
        assert 0, "what u doing boi"


def generate_string(constraint, max_string_size=100):
    constraint_dict, min_size = constraint
    #sample a size from min to max
    size = random.randint(min_size, max_string_size)
    indices = set(range(size))
    slist = random.choices(printable[:-4] , k=size)
    # schematically:
    #print("min_size", min_size)
    #print("size", size)
    for item in constraint_dict:
        reg, preg = item.ree, item.pre
        num_to_insert = max(0, constraint_dict[item] - len(re.findall(reg, ''.join(slist))))
        if len(indices) < num_to_insert: return None
        indices_to_insert = set(random.sample(indices, k=num_to_insert))
      
        for i in indices_to_insert:
            slist[i] = preg.sample()
        indices = indices - indices_to_insert
    string = ''.join(slist)
    if len(string) > max_string_size: return string[:max_string_size] 
    return string



################################### UTILS #################################
# merge 2 constraint dictionaries together
# a constraint dictionary keep track of how many tokens are to be needed on input
def add_constr(c1, c2=None):
    if c2 is None:
        return c1
    d1, m1 = c1
    d2, m2 = c2
    min_size = max(m1, m2)
    d = defaultdict(int)
    for item in set(d1.keys()) | set(d2.keys()):
        d[item] = max(d1[item], d2[item])
    return d, min_size

def stepped_abs(x):
    return x + 1 if x >= 0 else abs(x)

if __name__ == '__main__':
    print ("hi i live")

    yo = GetSpan.generate()
    print (yo)

    print (yo.constr)
    input_str = generate_string(yo.constr)
    print (input_str)
    print (yo.str_execute(input_str))

    print ("get token ")
    get_tk = GetToken.generate()
    print (get_tk)
    input_str = generate_string(get_tk.constr)
    print (input_str)
    print (get_tk.str_execute(input_str))



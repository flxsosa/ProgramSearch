#import random
import pregex as pre
#import math



assert(pre.create("f(a|o)*") == pre.Concat([pre.String("f"), pre.KleeneStar(pre.Alt([pre.String("a"), pre.String("o")]))]))
assert(pre.create("fa|o*") == pre.Concat([pre.String("f"), pre.Alt([pre.String("a"), pre.KleeneStar(pre.String("o"))])]))
assert(pre.create("(f.*)+") == pre.Plus(pre.Concat([pre.String("f"), pre.KleeneStar(pre.dot)])))
assert(pre.create("a*foo&&", {"foo":pre.create("A"), "&&":lambda r:pre.KleeneStar(r)}) == pre.create("a*A*"))

test_cases = [
	("foo", "fo", False),
	("foo", "foo", True),
	("foo", "fooo", False),
	("foo", "fo*", True),
	("foo", "fo+", True),
	("foo", "f(oo)*", True),
	("foo", "f(a|b)*", False),
	("foo", "f(a|o)*", True),
	("foo", "fa|o*", True),
	("foo", "fo|a*", False),
	("foo", "f|ao|ao|a", True),
	("f"+"o"*50, "f"+"o*"*10, True),
	("f"+"o"*50, "fo" + "*"*10, True),
	("foo", "fo?+", True),
	("foo", "fo**", True),
	("(foo)", "\\(foo\\)", True),
	("foo foo. foo foo foo.", "foo(\\.? foo)*\\.", True),
	("123abcABC ", ".+", True),
	("123abcABC ", '\\w+', False),
	("123abcABC ", "\\w+\\s", True),
	("123abcABC ", "\\d+\\l+\\u+\\s", True)
]
for (string, regex, matches) in test_cases:
	print("\nParsing", regex)
	r = pre.create(regex)
	print("Matching", string, r)
	assert(matches == (r.match(string)>float("-inf")))

def matchFirst(left, right, string): #left|right should match string through the left path
    s1 = pre.Alt([left, right]).match(string) 
    s2 = pre.Alt([left, pre.String("")]).match(string)
    print("\nTesting", string, "on", str(left), "vs", str(right))
    assert(s1==s2) 
matchFirst(pre.create("." + "0"*99), pre.create("\d"*100), "0"*100)
matchFirst(pre.create("(a|b)0"), pre.create("a."), "a0")

print("\nTesting brackets")
try:
    r = pre.create("(foo")
    assert(False)
except pre.ParseException: pass
try:
    r = pre.create("foo)")
    assert(False)
except pre.ParseException: pass

print("Testing natural frequencies")
print("hello")
assert(pre.create("\l*", natural_frequencies=True).match("hello") > pre.create("\l*").match("hello"))
print("jjjj")
assert(pre.create("\l*", natural_frequencies=True).match("jjjj") < pre.create("\l*").match("jjjj"))

print("Testing bigram")
c = pre.CharacterClass("abc", ps={"":[1/3, 1/3, 1/3], "a":[0.8,0.1,0.1], "b":[0.1,0.8,0.1], "c":[0.1,0.1,0.8]})
r = pre.KleeneStar(c)
assert(r.match("aaaabbbbcccc") > r.match("abcabcabcabc"))

#class Foobar():
#	def sample(self, state=None):
#		if random.random() > 0.5:
#			return "foo"
#		else:
#			return "bar"
#
#	def match(self, string, state):
#		if string=="foo" or string=="bar":
#			return math.log(1/2), state + 1
#		else:
#			return float("-inf"), None
#foobar = pre.Wrapper(Foobar())

class Empty():
	def sample(self, state=None):
		return ""

	def match(self, string, state):
		if string=="":
			return 0, state
		else:
			return float("-inf"), None
empty = pre.Wrapper(Empty())

string = "foobar"

#regex = pre.create("%%%", {"%":foobar, "&":empty})
#print("Testing", string, regex)
#score, state = regex.match("foobar", state=0)
#assert(score == float("-inf"))

#regex = pre.create("%%", {"%":foobar, "&":empty})
#print("Testing", string, regex)
#score, state = regex.match("foobar", state=0)
#assert(score == 2 * math.log(1/2))
#assert(state == 2)

#regex = pre.create("%*", {"%":foobar, "&":empty})
#print("Testing", string, regex)
#score, state = regex.match("foobar", state=0)
#assert(score > float("-inf"))
#assert(state == 2)

#regex = pre.create("foo%", {"%":foobar, "&":empty})
#print("Testing", string, regex)
#score, state = regex.match("foobar", state=0)
#assert(score > float("-inf"))
#assert(state == 1)

#regex = pre.create("%*&", {"%":foobar, "&":empty})
#print("Testing", string, regex)
#score, state = regex.match("foobar", state=0)
#assert(score > float("-inf"))

#Test save/load:
import pickle
import os
r = pre.create("\\d*|foo?|.+")
print(r, repr(r))
assert(r==r)
with open('regex_Test.p', 'wb') as file:
	pickle.dump(r, file)
with open('regex_Test.p', 'rb') as file:
	r2 = pickle.load(file)
	print(r2, repr(r2))
	assert(r == r2)
os.remove("regex_Test.p")

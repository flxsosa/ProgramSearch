Abstraction TODO:

THURSDAY:

- [X] figure out how to load my stuff --resume
- [X] train an easy value fun
	for now, R will use:
	spec.abstract() == p ... not sure if good!!

- [ ] begin to work on better abstraction
	- [ ] modules?? - seems like an optimization at this point ...

- [ ] better reward fn
	- [ ] simplification
	- [ ] online gradient descent

- [ ] hack to use noExecution policy but AbstractPointerNet distance?

- [ ] fix symmetry problem 
		- summation issue???
		- reward is off ... 

- [ ] why is AbstractPointerNet worse than noExecution?
	- hyp1: just bc of 
	- hyp2: only having a decoder makes it train better




********
Understand high level structure of code
	-[X] what distinguishes noExecution rendering?? - ln 774, embedding a symbolsequence (grounds out in self.model which is a GRU)

- [X] implement abstracted DSL

- [X] implement abtractify fn

- [ ] understand why one version of p has high beam score while other has low

- NoExecution model seems not to have a formal objectEncoder ... this could be problematic
- [X] Train NoExecution model

Train abstraction modules!!
- there is an R function which i may need to hack to get my differentiable renderer going ... 

Implement abstract noExecution model

Need to add to noExecution:
	- [ ] distance fn
	- [ ] proper objectEncoder
		easy mode:
		- [ ] just use a standard text - based encoder
		hard mode:
		- [ ] for each dsl element, model should have an abstract module for encoding it. SHOULD THIS JUST BE THE OBJECT ENCODER for policy prediciton too??

Subclass+modify a2c.py so it is abstract
	modify scopeEncoding or don't use it ...
	add an object encoder?
	forwardsample.batchedRollout??






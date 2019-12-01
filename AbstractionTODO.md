Abstraction TODO:


- nonmodular seems to beat modular. could be because of specEncoding as rnn init.
	- [X] test theory by retraining nonmodular wihtout specEncoding as rnn init
		- [ ] add this to testing
	- [ ] see if the effect goes away for harder problems + longer time - running now
	- [ ] find a way to incorporate that goodness into modular version --- at the root? should think about this

- [ ] triplet loss
- [ ] bias experiment
- [ ] e-graph


- [X] test out graphs.py
- [X] modify graphs.py so it has the correct names
- [X] modify driver.py so it can name experimentOutput subdirs
- [X] make test_and_plot.py script 
- [X] test full REPL comparison
- [X] add timestamping to new script so it doesn't overwrite
- [X] run a bunch of new tests with more objects and perhaps more time?
- [X] make the plot I wanted



##
- [ ] kevin reweights policy RL so that it doesn't give more score to easier things by normalizing by number of hit trajectories.
- [ ] it seems that there is a small bug in kevin's code, he sums value losses but averages policy losses in a batch.


- [ ] implement contrastive divergence/triplet loss
	- [ ] quadruple triplet loss ...
	- [ ] figure out how to get it in code
	- [ ] probably recoding A2C
	- [ ] question: is .toTrace() the same as matching the program graph?? if so, then I can feel fine about the RL loss


	- [ ] writing sampling code?

Building:
- [X] build out proper eval so I can compare super easily!!!!!

- [ ] train value with noExecution - 
- [ ] test noexecution with value? - what does this even look like ?

- [ ] make the final reward fn!!!

- [ ] differentiable rendering:
	- [ ] fix the off by one errors of the sigmoid
	- [ ] find a cleaner way to thread params through the system
	- [X] implement get_param_count
	- [X] implement get and set params
	- [X] put limits on differentiating so nothing goes out of bounds - clamp
	- [ ] implement something more sophisticated so rectangles and such have reasonable sizes
	- [ ] implement random sampling

- [ ] better reward fn
	- [ ] simplification - will ask Eric
	- [ ] random sampling?
	- [X] prettyPrint

- [ ] begin to work on better abstraction
	- [ ] modules?? - seems like an optimization at this point ...

- [ ] hack training to use noExecution policy but AbstractPointerNet distance? - seems like an optimization at this point

- [X] proper train and test situation??

- [X] fix symbolic symmetry bug in training - prettyPrint
- [ ] fix nn symmetry bug - positional encoding??? - kinda a bad hack ...

- [X] retrain critic with this new info

- [ ] compare with non-abstract version
- [ ] find a way to test on IoU for real

- [X] train a normal network - in imitation phase now


##FRIDAY:

1) scale up to compare to CAD fully - should get testing fully set up.
	a) compare dumb abstraction with good abstraction.
	b) Mess with value function objective to determine correct form.


#Real RL question: what is the difference between rollout training and something hacky w swapping or contrast?
	- RL is probably the correct unbiased situation i want for my search algo


First goal: distinguish structure and params
	Can i compare well with kevin's CAD synthesis when using no REPL?
	- 	form of value fun & how to train?

Second goal: no struct + param distinction 
	- "any way search"
	- what does reward fn look like??? What is policy supervision?


Formulating+testing:
- [ ] can we learn to successfully judge sketches?
- [ ] what is the right way to train a sketch judger?
- [ ] what is the correct form of a sketch judger?

ways to train value fun:
- [ ] Simplest, maybe best: Reinforce on prob of correctness. 
	- initial observation - doesn't seem to change much over the course of rollout ... is basically a judge of hardness ...
- [ ] train on "if it is possible?" - doesn't mean anything on program graph bc you can always start over. Can we do it on individual program tree?
	What did armando say: "top down better for big programs", ""


Debugging:
- [ ] fix symmetry issue



***
Other goals:

- [ ] fix symmetry problem + scoping issues
		- summation issue???
		- reward is off ... 
		- what are the non-parses in the sampling?
		- can try beam to make myself feel better
		- are objectencodings converging to zero? - no, dont think so
		- are objectencodings actually working? check eq - seems so ... 
		- ask kevin: does graph rename and reorder vars for viewing pleasure?
		- my thing breaks programGraph. idk when program graph is used tho

- [ ] why is AbstractPointerNet worse than noExecution?
	- hyp1: just bc of hidden state
	- hyp2: only having a decoder makes it train better



THURSDAY:
- [X] figure out how to load my stuff --resume
- [X] train an easy value fun
	for now, R will use:
	spec.abstract() == p ... not sure if good!!


Test currently training system!!
- [X] build simple demo script


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






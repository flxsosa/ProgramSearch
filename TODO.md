# TODO THINGS

## CODE TO DO, Wed:
- [X] test generation of multiple const
	- [ ] determine correct distribution for constant
- [X] test '-' and space delims
- [ ] finetune RL
- [ ] single network, value and policy head
- [ ] train with these new fixes on GCP

- [ ] do proper robustfill baseline
	- [ ] refactor beam
	- [ ] throw out invalid progs as they go

- [ ] save version of the data






## ROBUST FILL REAL TRAINING TESTING
- [X] Parallelize Data Generation
- [X] Formalise Train / Test Scaffold

## ROBUST FILL ABLATION STUDIES 
- [X] With / Without Intermediates (with / without scratch) (maybe lower priority)
- [X] With / Without Value Function 
- [ ] Normal RobustFill (RNN)
- [X] Robustfill which renders and then commits for each line (like Xinyun and Kevin) (also maybe lower priority)
- [X] Beam Search vs A* (pending . . . )
- [X] SMC
- [X] MCTS

## OTHER DOMAINS
- [ ] Numpy Manipulations
- [ ] ??? With Long Programs
- [ ] CAD -- figure out situation there

## WRITING/PITCHING
- [X] write for ICML workshop paper
- [ ] Ask advisors for help pitching our approach

Kevin's input: let's focus more on formalizing the MDP situation and pushing on the search + value function additions, which are what seperate us from prior work
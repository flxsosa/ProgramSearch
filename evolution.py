from programGraph import ProgramGraph
from API import *
import time
import numpy as np
from utilities import getArgument

class Evolution(Solver):

	def __init__(self, DSL, pop_size=1):
		self.population = Population(pop_size)
		self.DSL = DSL

	def _infer(self, spec, loss, timeout):
		# Start inference
		start_time = time.time()

		best_loss = None

		# Run inference
		while time.time() - start_time < timeout:
			
			# Traverse population and mutate genomes
			for idx in range(len(self.population.pop)):
				
				# Mutate genome
				self.population.pop[idx] = self.mutate(self.population.pop[idx])

				# Obtain loss
				loss = self._loss(self.population.pop[idx])

				# Keep record of best genome
				if best_loss is None or best_loss > loss:
					best_loss = loss
					self.population.best_genome = self.population.pop[idx]

			# Report best solution so far
			self._report(self.population.best_genome)
			# print(self.population.best_genome.prettyPrint())

	def mutate(self, genome):
		'''
		Mutates the Program Graph according to its lexicon. Assuming
		lexicon is 2D CSG programs for now.

		TODO: Abstract this code to allow for any lexicon and 
				mutation distribution
		'''

		# Distribution over mutation (currently just uniform)
		mutation_distribution = [1.0/len(self.DSL.operators)]*len(self.DSL.operators)

		# Sample from DSL
		operator = np.random.choice(self.DSL.operators, p=mutation_distribution)
		tp = operator.type
		object = None

		# Check if operator is an arrow
		if not tp.isArrow:
			object = operator()

		# If it is, supply it with arguments
		else:
			arguments = [getArgument(t, genome) for t in tp.arguments]

			if not any(a is None for a in arguments):
				try:
					object = operator(*arguments)
				except:
					pass
		
		# Return new, mutated program graph
		if object is None: 
			return self.mutate(genome)
		
		return genome.extend(object)

class Population():

	def __init__(self, size, elitism=1, state=None):
		'''
		Class for populations of Program Graphs. 

		key:     population key
		size:    population size
		elitism: number of members that must be passed from previous 
				 gen to next gen
		'''
		self.size = size
		self.best_genome = None
		self.last_best = 0
		self.elitism = elitism
		self.pop = [ProgramGraph([]) for _ in range(self.size)]

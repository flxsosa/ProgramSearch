#neural network for ROBUT
import torch
# Text text processing library and methods for pretrained word embeddings
from torch import nn
import numpy as np

# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor

# args = {
# 	num_actions : len(actions)
# }

#we will use namedtensor because it should help with attention ...

#pip install -q torch torchtext opt_einsum git+https://github.com/harvardnlp/namedtensor

def states_to_tensors(x):
	"""
	assumes x is a list of states. This is the nastiest part
	masks will be batch x Examples x strLen x inFeatures (inFeatures=8)
	chars will be batch x Examples x strLen x stateLoc (stateLoc=4)
	last_butts will be batch (and the entries will be longs)
	"""	
	#chars:
	inputs, scratchs, committeds, outputs, masks, last_butts = zip(*x)

	inputs = np.stack( [i for i in inputs])
	in_tensor = ntorch.tensor(inputs, ("batch", "Examples", "strLen"))

	scratchs = np.stack( scratchs)
	scratch_tensor = ntorch.tensor(scratchs, ("batch", "Examples", "strLen"))

	committeds = np.stack(committeds)
	commit_tensor = ntorch.tensor(committeds, ("batch", "Examples", "strLen"))

	outputs = np.stack(outputs)
	out_tensor = ntorch.tensor(outputs, ("batch", "Examples", "strLen"))

	chars = ntorch.stack([in_tensor, out_tensor, commit_tensor, scratch_tensor], 'stateLoc')
	chars = chars.transpose("batch", "Examples", "strLen", "stateLoc")
	
	#masks:
	masks = np.stack(masks)
	masks = ntorch.tensor( masks, ("batch", "Examples", "inFeatures", "strLen"))
	masks = masks.transpose("batch", "Examples", "strLen", "inFeatures")
	
	last_butts = np.stack(last_butts)
	last_butts = ntorch.tensor(last_butts, ("batch", "extra")).sum("extra")

	return chars, masks, last_butts


class AttnPooling(nn.Module):
	pass

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.char_embedding = ntorch.nn.Embedding(
								args.num_char_types, args.char_embed_dim
									).spec("charEmb")

		self.column_encoding = ntorch.nn.Linear( 
								4*args.char_embed_dim+8, args.column_encoding_dim
									).spec("inFeatures", "E") #TODO
		self.other = None#TODO



	def forward(self, x):
		chars, masks, last_butt = states_to_tensors(x)
		charEmb = self.char_embedding(chars)
		charEmb.stack(inFeatures=('charEmb', 'stateLoc'))
		x = ntorch.cat([charEmb, masks], "inFeatures")
		e = self.column_encoding(x)

		h = self.other(e) #maybe attention, maybe a dense block, whatever

		#h should have dims batch x Examples x hidden -- 
		return h


class Model(nn.Module):
	def __init__(self, args):
		super(Model, self).__init__()
		self.encoder = Encoder(args)

		#pooling:
		if args.pooling == "max":
			self.pooling = lambda x: x.max("examples")
		elif args.pooling == "mean":
			self.pooling = lambda x: x.mean("examples")
		elif args.pooling == "attn":
			self.pooling = AttnPooling(args)

		else: assert 0, "oops, attention is wrong"

		self.action_decoder = ntorch.nn.Linear(args.h_out, args.num_actions).spec("h", "actions")

	def forward(self, x): #x is a state or a batch of states
		x = self.encoder(x)
		x = self.pooling(x)
		return self.action_decoder(x) #TODO, this may not exactly be enough?

	def learn_supervised(self, states, actions):
		raise NotImplementedError

	def sample_action(self, state):
		raise NotImplementedError

	def save(self, loc):
		torch.save(self.state_dict(), loc)

	def load(self, loc):
		self.load_state_dict(torch.load(loc))




class Agent:
	def __init__(self, actions):
		self.actions = actions

		if torch.cuda.is_available():
			self.nn = Model(args).cuda() #TODO args
		else:
			self.nn = Model(args)

	def act(self, state):
		return self.nn.sample_action(state)

	# not a symbolic state here
	# actions are 2, 3 instead of 0,1 index here
	def learn_supervised(self, states, actions):
		loss = self.nn.learn_supervised(states, actions)
		return loss

	def save(self, loc):
		self.nn.save(loc)

	def load(self, loc):
		self.nn.load(loc)

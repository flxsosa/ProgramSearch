#neural network for ROBUT
import torch
# Text text processing library and methods for pretrained word embeddings
from torch import nn
import numpy as np
import args
# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
import torch.nn.functional as F
#we will use namedtensor because it should help with attention ...
#pip install -q torch torchtext opt_einsum git+https://github.com/harvardnlp/namedtensor
from collections import namedtuple
TraceEntry = namedtuple("TraceEntry", "prev_s action reward s done")


class AttnPooling(nn.Module):
	"""
	attention pooling over examples
	"""
	pass

class DenseLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(DenseLayer, self).__init__()
        self.linear = ntorch.nn.Linear(input_size, output_size).spec("h", "h")
        #self.activation 
    def forward(self, x):
        return self.linear(x).relu()


class DenseBlock(nn.Module):
    def __init__(self, num_layers, growth_rate, input_size, output_size):
        super(DenseBlock, self).__init__()

        modules = [DenseLayer(input_size, growth_rate)]
        for i in range(1, num_layers - 1):
            modules.append(DenseLayer(growth_rate * i + input_size, growth_rate))
        modules.append(DenseLayer(growth_rate * (num_layers - 1) + input_size, output_size))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        inputs = [x]
        for layer in self.layers:
            output = layer(ntorch.cat(inputs, "h"))
            inputs.append(output)
        return inputs[-1]

class Encoder(nn.Module):
	def __init__(self):
		super(Encoder, self).__init__()

		self.char_embedding = ntorch.nn.Embedding(
								args.num_char_types, args.char_embed_dim
									).spec("stateLoc", "charEmb") #TODO: no idea if this is good

		if args.column_enc == 'linear':
			self.column_encoding = ntorch.nn.Linear( 
									4*args.char_embed_dim+7, args.column_encoding_dim
										).spec("inFeatures", "E") #TODO
		elif args.column_enc == 'conv':
			self.column_encoding = ntorch.nn.Conv1d(
			in_channels=4*args.char_embed_dim+7,
			out_channels=args.column_encoding_dim,
			kernel_size=(args.kernel_size),
			padding=(args.kernel_size-1)/2).spec("inFeatures", "strLen", "E")
		else: assert 0

		print("WARNING: there are only 7 masks?? Change this in robut_net.py")

		if args.encoder == 'dense':

			self.MLP = DenseBlock(args.num_dense_layers, args.growth_rate, args.column_encoding_dim*args.strLen, args.h_out) #maybe attention, maybe a dense block, whatever

		elif args.encoder == 'transformer':
			assert 0, "didnt write yet"


	def forward(self, chars, masks, last_butt):
		#chars, masks, last_butt = states_to_tensors(x)
		charEmb = self.char_embedding(chars)
		charEmb = charEmb.stack(('charEmb', 'stateLoc'), 'inFeatures')

		x = ntorch.cat([charEmb, masks], "inFeatures")
		e = self.column_encoding(x)
		#TODO: incorporate last_button here, using a repeat function, possibly ...
		#e = ntorch.cat([e, last_butt], "batch")

		if args.encoder =='dense':
			e = e.stack( ('strLen', 'E'), 'h')
			#incorporate last_butt?
		h = self.MLP(e) #maybe attention, maybe a dense block, whatever
		#h should have dims batch x Examples x hidden -- 
		return h


class Model(nn.Module):
	def __init__(self, num_actions, value_net=False):
		super(Model, self).__init__()
		self.encoder = Encoder()

		self.button_embedding = ntorch.nn.Embedding(
								num_actions+1, args.button_embed_dim
									).spec("batch", "h")
		self.fc = ntorch.nn.Linear(args.h_out+args.button_embed_dim, args.h_out)
		#pooling:
		if args.pooling == "max":
			self.pooling = lambda x: x.max("Examples")
		elif args.pooling == "mean":
			self.pooling = lambda x: x.mean("Examples")
		elif args.pooling == "attn":
			self.pooling = AttnPooling(args)
		else: assert 0, "oops, attention is wrong"

		if value_net:
			assert 0, "didnt write it yet, but it's really simple"

		self.action_decoder = ntorch.nn.Linear(args.h_out, num_actions).spec("h", "actions")
		self.lossfn = ntorch.nn.CrossEntropyLoss().spec("actions") #TODO
		self.lossfn.reduction = None #TODO XXX FIXME DON"T LEAVE THIS
		self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

	def forward(self, chars, masks, last_butts): 
		x = self.encoder(chars, masks, last_butts)
		x = self.pooling(x)

		lb_emb = self.button_embedding(last_butts)
		x = ntorch.cat([x, lb_emb], "h")
		x = self.fc(x).relu()
		x = self.action_decoder(x) #TODO, this may not exactly be enough?
		# x = x._new(
		#  F.log_softmax(x._tensor, dim=x._schema.get("actions"))
		# 	) #TODO XXX FIXME DON"T LEAVE THIS
		return x 

	def learn_supervised(self, chars, masks, last_butts, targets):
		self.train()
		self.opt.zero_grad()

		output_dists = self(chars, masks, last_butts)
		loss = self.lossfn(output_dists, targets)
		loss.backward()
		self.opt.step()
		return loss

	def sample_action(self, chars, masks, last_butts):
		self.nn.eval()
		raise NotImplementedError

	def save(self, loc):
		torch.save(self.state_dict(), loc)

	def load(self, loc):
		self.load_state_dict(torch.load(loc))

class Agent:
	def __init__(self, actions, use_cuda=None, value_net=False):
		self.actions = actions
		self.idx = {x.name: i for i, x in enumerate(actions)}
		self.name_to_action = {x.name: x for x in actions}
		self.idx_to_action = {self.idx[x.name]: self.name_to_action[x.name] for x in actions} 

		self.use_cuda = use_cuda
		self.value_net = value_net
		if use_cuda == None: self.use_cuda = torch.cuda.is_available()

		if self.use_cuda:
			self.nn = Model(len(actions)).cuda() #TODO args
			if self.value_net: self.Vnn = Model(len(actions), value_net=True).cuda()
		else:
			self.nn = Model(len(actions))
			if self.value_net: self.Vnn = Model(len(actions), value_net=True)

	def states_to_tensors(self, x):
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
		chars = chars.transpose("batch", "Examples", "strLen", "stateLoc").long()
		
		#masks:
		masks = np.stack(masks)
		masks = ntorch.tensor( masks, ("batch", "Examples", "inFeatures", "strLen"))
		masks = masks.transpose("batch", "Examples", "strLen", "inFeatures").float()
		
		last_butts = np.stack(last_butts)
		last_butts = ntorch.tensor(last_butts, ("batch", "extra")).sum("extra").long()

		if self.use_cuda:
			return chars.cuda(), masks.cuda(), last_butts.cuda()
		else:
			return chars, masks, last_butts

	def actions_to_target(self, actions):
		indices = [self.idx[a.name] for a in actions] 
		target = ntorch.tensor( indices, ("batch",) ).long()
		return target.cuda() if self.use_cuda else target

	def sample_actions(self, states):
		#assumes list of states, returns corresponding list of actions
		chars, masks, last_butts = self.states_to_tensors(states)
		
		logits = self.nn.forward(chars, masks, last_butts)
		dist = ntorch.distributions.Categorical(logits=logits, dim_logit="actions")
		sample = dist.sample()
		action_list = [self.idx_to_action[sample[{"batch":i}].item()] for i in range(sample.shape["batch"])]
		return action_list

	def act(self, state):
		a_list = self.sample_actions([state])
		assert len(a_list) == 1
		return a_list[0]

	def best_actions(self, states):
		#assumes list of states, returns corresponding list of actions
		#TODO for top k actions, use _, argmax = logits.topk("actions", k)
		chars, masks, last_butts = self.states_to_tensors(states)
		logits = self.nn.forward(chars, masks, last_butts)
		_, argmax = logits.max('actions')
		action_list = [self.idx_to_action[argmax[{"batch":i}].item()] for i in range(argmax.shape["batch"])]
		return action_list

	def topk_actions(self, states, k):
		#assumes list of states, returns list of lists of k actions
		#TODO for top k actions, use _, argmax = logits.topk("actions", k)
		# returns a tuple of button, score
		chars, masks, last_butts = self.states_to_tensors(states)
		logits = self.nn.forward(chars, masks, last_butts)
		#lls = logits.log_softmax("actions")
		lls = logits._new(
		 F.log_softmax(logits._tensor, dim=logits._schema.get("actions"))
			) #TODO XXX FIXME DON"T LEAVE THIS
		ll, argmax = lls.topk('actions', k)
		action_list = [[ (self.idx_to_action[argmax[{"batch":i, "actions":kk}].item()], ll[{"batch":i, "actions":kk}].item()) for kk in range(k)] for i in range(argmax.shape["batch"])  ] 
		return action_list

	def all_actions(self, states, k):
		#assumes list of states, returns list of lists of k actions
		#TODO for top k actions, use _, argmax = logits.topk("actions", k)
		# returns a tuple of button, score
		chars, masks, last_butts = self.states_to_tensors(states)
		logits = self.nn.forward(chars, masks, last_butts)
		#lls = logits.log_softmax("actions")
		lls = logits._new(
		 F.log_softmax(logits._tensor, dim=logits._schema.get("actions"))
			) #TODO XXX FIXME DON"T LEAVE THIS
		ll, argmax = lls.topk('actions', k)
		action_list = [[ (self.idx_to_action[argmax[{"batch":i, "actions":kk}].item()], ll[{"batch":i, "actions":kk}].item()) for kk in range(k)] for i in range(argmax.shape["batch"])  ] 
		return action_list

	# not a symbolic state here
	# actions are 2, 3 instead of 0,1 index here
	def learn_supervised(self, states, actions):
		chars, masks, last_butts = self.states_to_tensors(states)
		targets = self.actions_to_target(actions)
		loss = self.nn.learn_supervised(chars, masks, last_butts, targets)
		return loss

	def get_rollouts(self, env, n_rollouts=1000, max_iter=30):


		from ROBUT import ROBENV
		s = env.reset()
		envs = []
		for _ in range(n_rollouts):
			e = ROBENV(env.inputs, env.outputs)
			e.reset()
			envs.append(e)

		traces = [ [] for _ in range(n_rollouts) ]
		for i in range(max_iter):
			if i==0:
				active_states = [s for _ in range(n_rollouts)]
			else:
				active_states = [t[-1].s for t in traces if not t[-1].done]
			action_list = self.sample_actions(active_states) if active_states else []
			#prevents nn running on nothing 
			action_list_iter = iter(action_list)
			active_states_iter = iter(active_states)
			if action_list == []: return traces

			for j in range(n_rollouts):
				if i>0 and traces[j][-1].done: #if done:
					continue
				a = next(action_list_iter)
				ss, r, done = envs[j].step(a)
				if i==0:
					prev_s = s
				else:
					prev_s = traces[j][-1].prev_s
					# xx = next(active_states_iter)
					# assert np.all(xx[0] == prev_s[0])
					# assert np.all(xx[1] == prev_s[1])
					# assert np.all(xx[2] == prev_s[2])
					# assert np.all(xx[3] == prev_s[3])
					# assert np.all(xx[4] == prev_s[4])
					# assert xx[5] == prev_s[5]
				traces[j].append( TraceEntry(prev_s, a, r, ss, done) )

		return traces

	def beam_rollout(self, env, beam_size=1000, max_iter=30):
		BeamEntry = namedtuple("BeamEntry", "env trace score")
		#TraceEntry = namedtuple("TraceEntry", "prev_s action reward s done")

		beam = []
		s = env.reset()
		top_actions = self.topk_actions([s], len(self.actions)) #the whole thing


		for action, score in top_actions[0]: #for first round:
			e = env.copy()
			ss, r, done = e.step(action)
			t = TraceEntry(s, action, r, ss, done)
			beam.append(BeamEntry(e, [t], score))

		beam = sorted(beam, key=lambda x: -x.score)


		#beam = ( env, trace, score )

		#get_done = lambda x: x.trace.done
		#get_score = lambda x: x.score

		for i in range(1, max_iter):
			#get top_k actions
			new_beam = []
			s_list = []
			for b in beam:
				if b.trace[-1].done:
					# if b.trace[-1].reward == 1: #if 
					# 	new_beam.append(b)
					continue
				#tr = b.trace
				s = b.trace[-1].s
				s_list.append(s)

			top_actions = self.topk_actions(s_list, len(self.actions))
			top_actions_iter = iter(top_actions)
			s_list_iter = iter(s_list)
			for b in beam:
				
				if b.trace[-1].done:
					if b.trace[-1].reward == 1: #if 
						new_beam.append(b)
					continue
				tr = b.trace
				#s = b.trace[-1].s

				_top_actions = next(top_actions_iter)
				s = next(s_list_iter)
				for action, score in _top_actions:
					e = b.env.copy()
					ss, r, done = e.step(action)
					if done and r < 1.0: #ie, if it is a crash state, don't use it
						continue
					tr.append( TraceEntry(s, action, r, ss, done) )
					new_beam.append(BeamEntry(e, tr, score + b.score))

				new_beam = sorted(new_beam, key=lambda x: -x.score)[:beam_size]
			new_beam = sorted(new_beam, key=lambda x: -x.score)[:beam_size]
			beam = new_beam
		return beam

	def save(self, loc):
		self.nn.save(loc)
		if self.value_net: 
			self.Vnn.save(loc+'vnet')

	def load(self, loc):
		self.nn.load(loc)
		if self.value_net: 
			self.Vnn.load(loc+'vnet')


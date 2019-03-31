#neural network for ROBUT
import torch
# Text text processing library and methods for pretrained word embeddings
from torch import nn
import numpy as np
import arguments.args as args
# Named Tensor wrappers
from namedtensor import ntorch, NamedTensor
import torch.nn.functional as F
#we will use namedtensor because it should help with attention ...
#pip install -q torch torchtext opt_einsum git+https://github.com/harvardnlp/namedtensor
from collections import namedtuple
TraceEntry = namedtuple("TraceEntry", "prev_s action reward s done")

class QEntry(object):
    def __init__(self, priority, env):
        self.score = priority
        self.env = env
    def __cmp__(self, other):
        return cmp(-self.score, -other.score)

    def __lt__(self, other):
        return -self.score < -other.score

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
        self.value_net = value_net
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
            #assert 0, "didnt write it yet, but it's really simple"
            self.action_decoder = ntorch.nn.Linear(args.h_out, 2).spec("h", "value")
            self.lossfn = ntorch.nn.NLLLoss().spec("value") #TODO
            self.lossfn.reduction = None #TODO XXX FIXME DON"T LEAVE THIS
        else:
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
        x = self.action_decoder(x) #TODO, this may not exactly be enough? also, wrong name
        if self.value_net:
            x = x._new(
                F.log_softmax(x._tensor, dim=x._schema.get("value"))
                ) #TODO XXX FIXME DON"T LEAVE THIS
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

    def rewards_to_target(self, rewards):
        target = ntorch.tensor( rewards, ("batch",) ).long().relu()
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
        if self.use_cuda:
            ll = ll.cpu()
            argmax = argmax.cpu()
        action_list = [[ (self.idx_to_action[argmax[{"batch":i, "actions":kk}].item()], ll[{"batch":i, "actions":kk}].item()) for kk in range(k)] for i in range(argmax.shape["batch"])  ] 
        return action_list

    def all_actions(self, states):
        #assumes list of states, returns list of lists of k actions
        #TODO for top k actions, use _, argmax = logits.topk("actions", k)
        # returns a tuple of button, score
        chars, masks, last_butts = self.states_to_tensors(states)
        logits = self.nn.forward(chars, masks, last_butts)
        #lls = logits.log_softmax("actions")
        lls = logits._new(
         F.log_softmax(logits._tensor, dim=logits._schema.get("actions"))
            ) #TODO XXX FIXME DON"T LEAVE THIS
        ll, argmax = lls.topk('actions', len(self.actions))
        if self.use_cuda:
            ll = ll.cpu()
            argmax = argmax.cpu()
        #action_list = [[ (self.idx_to_action[argmax[{"batch":i, "actions":kk}].item()], ll[{"batch":i, "actions":kk}].item()) for kk in range(k)] for i in range(argmax.shape["batch"])  ] 
        return ll, argmax

    # not a symbolic state here
    # actions are 2, 3 instead of 0,1 index here
    def learn_supervised(self, states, actions):
        chars, masks, last_butts = self.states_to_tensors(states)
        targets = self.actions_to_target(actions)
        loss = self.nn.learn_supervised(chars, masks, last_butts, targets)
        return loss

    def value_fun_optim_step(self, states, rewards):
        chars, masks, last_butts = self.states_to_tensors(states)
        targets = self.rewards_to_target(rewards)
        print("TARGETS",targets.sum("batch").item())
        loss = self.Vnn.learn_supervised(chars, masks, last_butts, targets)
        return loss

    def compute_values(self, states):
        chars, masks, last_butts = self.states_to_tensors(states)
        #self.Vnn.eval()
        output_dists = self.Vnn(chars, masks, last_butts)
        return output_dists

    def get_rollouts(self, initial_envs, n_rollouts=1000, max_iter=30):
        """
        initial_envs is a list of initial envs
        n_rollouts is per initial_env
        """
        from ROBUT import ROBENV
        n_initial_envs = len(initial_envs)
        envs = []
        traces = []
        active_states = []
        for env in initial_envs:
            env.reset()
            for _ in range(n_rollouts):
                e = env.copy()
                envs.append(e)
                traces.append([])
                active_states.append( env.last_step[0] )

        #traces = [ [] for _ in range(n_rollouts) ]
        for i in range(max_iter):
            if not i==0:
                active_states = [t[-1].s for t in traces if not t[-1].done]
            action_list = self.sample_actions(active_states) if active_states else []
            #prevents nn running on nothing 
            action_list_iter = iter(action_list)
            active_states_iter = iter(active_states)
            if action_list == []: return traces

            for j in range(n_initial_envs*n_rollouts):
                if i>0 and traces[j][-1].done: #if done:
                    continue
                a = next(action_list_iter)
                ss, r, done = envs[j].step(a)
                if i==0:
                    prev_s = envs[j].last_step[0]
                else:
                    prev_s = traces[j][-1].prev_s
                traces[j].append( TraceEntry(prev_s, a, r, ss, done) )
        return traces

    def beam_rollout(self, env, beam_size=1000, max_iter=30, verbose=False):
        BeamEntry = namedtuple("BeamEntry", "env score")
        env.reset()
        beam = [ BeamEntry(env.copy(), 0.0)] #for _ in range(beam_size)]
        solutions = []
        for t in range(max_iter):
            print("beam iteration", t)
            state_list = [be.env.last_step[0] for be in beam]
            # get the current log-likelihood for extending the head of the beam
            lls, argmax = self.all_actions(state_list)
            vals

            lls = lls.detach().cpu().numpy()
            # massage the previous into the right shape and add it to the current head
            lls_prev = np.array([be.score for be in beam])
            curr_beam_len = len(beam)
            lls_prev = np.expand_dims(lls_prev, axis=1)
            lls_prev = np.repeat(lls_prev, len(self.actions), axis=1)
            # this is the llhood of all the candidate beams which we need to sort
            lls_curr = lls_prev + lls
            lls_curr = np.reshape(lls_curr, (curr_beam_len * len(self.actions), ))

            idxs = np.argsort(-lls_curr)
            new_beam = []
            new_beam_size = 0
            for idx in idxs:
                s_a_idx = np.unravel_index(idx, (curr_beam_len, len(self.actions)))
                action = self.idx_to_action[argmax[{"batch": int(s_a_idx[0]), "actions": int(s_a_idx[1])}].item()]
                state = beam[int(s_a_idx[0])].env.copy()

                state.step(action)
                if state.last_step[1] == -1: #reward
                    if verbose: print ("crasherinoed!")
                    continue
                if state.last_step[1] == 1: #reward
                    solutions.append(state)
                    if verbose: print ("success ! ")
                    continue

                bentry = BeamEntry(state, lls_curr[idx])
                new_beam.append( bentry )
                new_beam_size += 1

                if new_beam_size >= beam_size:
                    break
                # print(action)
                # print(state)
                # print (bentry.env.last_step)

            beam = new_beam                
        return beam, solutions 

    def a_star_rollout(self, env, batch_size=1000, max_count=1000*723*30, max_iter=30, verbose=False):

        from queue import PriorityQueue
        q = PriorityQueue()
        env.reset()
        q.put(QEntry(0.0, env))

        solutions = []
        while not q.empty() and q.qsize() < max_count: #TODO
            toExpand = [ q.get_nowait() for _ in range(min(batch_size, q.qsize()))]
            state_list = [ entry.env.last_step[0] for entry in toExpand]

            action_ll, argmax = self.all_actions(state_list) 
            value_ll = self.compute_values(state_list)
            print('ran both nns')   
            for i, entry in enumerate(toExpand):
                score = entry.score
                e = entry.env
                state_ll = value_ll[{"batch":i, "value":1}].item() #TODO XXX
                for idx in range(len(self.actions)): #can change this later to filter/prune
                    action = self.idx_to_action[argmax[{"batch": i, "actions": idx}].item()]
                    e_new = e.copy()

                    e_new.step(action)
                    if e_new.last_step[1] == -1: #reward
                        if verbose: print ("crasherinoed!")
                        continue
                    if e_new.last_step[1] == 1: #reward
                        solutions.append(e_new)
                        if verbose: print ("success ! ")
                        continue
                    if len(e_new.pstate.past_buttons) > max_iter:
                        if verbose: print("gone too far!")
                        continue

                    new_score = score + action_ll[{"batch": i, "actions": idx}].item() + state_ll
                    q.put(QEntry(new_score, e_new))


        return solutions




    def save(self, loc):
        self.nn.save(loc)
        if self.value_net: 
            self.Vnn.save(loc+'vnet')

    def load(self, loc):
        self.nn.load(loc)
        print(f"loaded policy net from {loc}")
        if self.value_net: 
            self.Vnn.load(loc+'vnet')
            print(f"loaded value net from {loc}")

if __name__ == '__main__':
    print ("hi")

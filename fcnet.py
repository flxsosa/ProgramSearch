"""
    the policy gradient agent
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

# ===================== FC NN CLASSIFIER =====================
if torch.cuda.is_available():
  def to_torch(x, dtype, req = False):
    tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x
else:
  def to_torch(x, dtype, req = False):
    tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x

class FCNet(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(FCNet, self).__init__()
        self.name = "FCNet"
        self.in_dim = in_dim
        h_dim = 100
        self.fc = nn.Linear(in_dim, h_dim)
        self.pred = nn.Linear(h_dim, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = x.view(-1, self.in_dim)
        x = F.relu(self.fc(x))
        x = F.log_softmax(self.pred(x), dim=1)
        return x

    def learn_supervised(self, state_batch, action_batch):
        X = to_torch(state_batch, "float")
        Y = to_torch(action_batch, "int")

        # optimize 
        self.opt.zero_grad()
        output = self(X)
        loss = F.nll_loss(output, Y)
        loss.backward()
        self.opt.step()

        return loss

class Agent:
    def __init__(self, input_dim, actions):
        self.actions = actions
        self.input_dim = input_dim

        if torch.cuda.is_available():
            self.nn = FCNet(input_dim, len(actions)).cuda()
        else:
            self.nn = FCNet(input_dim, len(actions))

    def act(self, state):
        state_torch = to_torch(np.array([state]), "float")
        
        action_logprob = self.nn(state_torch).detach().cpu().numpy()[0]
        best_action_idx = np.argmax(action_logprob) 
        return self.actions[best_action_idx]

    # not a symbolic state here
    # actions are 2, 3 instead of 0,1 index here
    def learn_supervised(self, states, actions):
        states = np.array(states)
        actions = np.array([self.actions.index(a) for a in actions])
        self.nn.learn_supervised(states, actions)


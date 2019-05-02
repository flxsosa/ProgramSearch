#train_rb_baseline.py


"""
TODO:
- [X] write scaffolding
- [X] test that it runs and trains

- [ ] allow env to be part of beam/forward sample process
- [ ] optimize beam

- [ ] testing code
"""
#from robut_net import Agent
#from load_args import args #requires

import torch
import time
from ROBUT import ALL_BUTTS
from ROB import get_supervised_sample, generate_FIO
from robut_data import get_supervised_batchsize, GenData, makeTestdata

from robustfill import RobustFill
import string

#big hack here 
LOAD_PATH = './models/rb_baseline.p'
SAVE_PATH = './models/rb_baseline.p'
ITERATIONS = 50000
BATCHSIZE = 4

PRINT_FREQ = 1
TEST_FREQ = 1
SAVE_FREQ = 10

DEBUG = True



input_vocabularies = ( string.printable[:-4], string.printable[:-4] )
target_vocabulary = ALL_BUTTS

def load_model():
    print(f"is cuda available? {torch.cuda.is_available()}")
    #model = RobustFill(*args, **kwargs) # TODO
    model = RobustFill(input_vocabularies, target_vocabulary, hidden_size=512, embedding_size=128, cell_type="LSTM", max_length=40)
    model.cuda()
    try:
        model.load(LOAD_PATH) # TODO, 
        #also not sure if i should just save the whole thing with model=torch.load(args.rb_load_path)
        print("loaded model")
    except FileNotFoundError:
        print ("no saved model found ... training from scratch")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("num params:", num_params)
    return model

def generate_rb_data(batchsize):
    specs = [] 
    As = []
    for _ in range(batchsize): 
        prog, inputs, outputs = generate_FIO(5)
        io = list(zip(inputs, outputs))
        specs.append(io)
        As.append( prog.flatten() )
    return specs, As

if DEBUG:
    debug_data = generate_rb_data(BATCHSIZE)

def train_model_supervised(model):
    enum_t2 = 0
    print_time = 0
    if not hasattr(model, 'train_iterations'): model.train_iterations = 0

    for i in range(ITERATIONS):
        specs, As = generate_rb_data(BATCHSIZE) if not DEBUG else debug_data

        # print(list(len( a) for a in As))
        # print(As[0])
        # print(specs[0])

        enum_t = time.time()

        t = time.time()
        loss = model.optimiser_step(specs, As)
        t2 = time.time()

        pt = time.time()
        if i%PRINT_FREQ == 0 and i!=0:
            print("iteration {}, loss: {:.5f}, network time: {:.5f}, gen samples time: {:.5f}, prev print time: {:.5f}, total other time: {:.5f}".format(model.train_iterations, loss, t2-t, enum_t - enum_t2, print_time, t-t3 ), flush=True)
        pt2 = time.time()
        print_time = pt2-pt

        t3 = t2
        if i%SAVE_FREQ == 0 and i!=0:
            model.save(SAVE_PATH)
            print("saved model", flush=True)
        if i%TEST_FREQ == 0 and i!=0:  
            print("testing...")
            specs, As = generate_rb_data(1)  if not DEBUG else debug_data[0][:1], debug_data[1][:1]
            print("TESTING SPEC:", specs)
            actions = model.sample(specs, n_samples=1)
            print("real actions:")
            print(As)
            print("model actions:")
            print(actions)
        enum_t2 = time.time()
    
        if hasattr(model, 'train_iterations'):
            model.train_iterations += 1
        else: model.train_iterations = 1
    model.save(SAVE_PATH)


if __name__ == '__main__':
    #load model or create model
    model = load_model()
    #train
    train_model_supervised(model)
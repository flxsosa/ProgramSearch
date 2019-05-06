
pooling = "max"

h_out = 128

import string
num_char_types = len(string.printable[:-4]) + 1

char_embed_dim = 20

column_encoding_dim = 32 #16

strLen = 100

encoder = 'dense'

num_dense_layers = 10

growth_rate = 128

button_embed_dim = 32

batchsize = 4000

train_iterations = 50000

parallel = True
n_processes = 8
save_path = 'models/new_RLfinetune.p'
load_path = 'models/new_RLfinetune.p'

encode_past_buttons = False
past_button_embed_dim = None
rnn_hidden_size = None

render_kind={'render_scratch' : 'yes',
			'render_past_buttons' : 'no'}
#big means fatter ... 10 layers of 128 growth rate

#Vnet_save_path = 'robut_vnet.p' 

print_freq = 2

save_freq = 100

test_freq = 1000

column_enc = 'conv' #'linear' #'conv' #'linear' 
kernel_size = 5

n_rollouts = 500

n_envs_per_rollout = 2

rl_iterations = 12000

rl_mode = 'both'

##test scaffold
test_type = 'smc'
use_value = True
resultsfile = './results/smc_val_new.p'
debug = False
use_prev_value = False

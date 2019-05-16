
pooling = "max"

h_out = 128

import string
num_char_types = len(string.printable[:-4]) + 1

char_embed_dim = 20

column_encoding_dim = 32 #16

strLen = 36

encoder = 'dense'

num_dense_layers = 10

growth_rate = 128

button_embed_dim = 128

batchsize = 4000

train_iterations = 50000

parallel = True
n_processes = 8
save_path = 'models/noscratch24k.p' #'robut_model_larger.p' #'robut_model_conv.p' # #'robut_model.p'
load_path = 'models/noscratch24k.p'

encode_past_buttons = True
past_button_embed_dim = 128
#rnn_hidden_size = 32 #I think the same as button_embed_dim
render_kind = {'render_scratch' : 'no',
			   'render_past_buttons' : 'yes'} #might not be what we want

#big means fatter ... 10 layers of 128 growth rate

#Vnet_save_path = 'robut_vnet.p' 

print_freq = 20

save_freq = 100

test_freq = 1000

column_enc = 'conv' #'linear' #'conv' #'linear' 
kernel_size = 5

n_rollouts = 40

n_envs_per_rollout = 25

rl_iterations = 12000

rl_mode = 'value only'

##test scaffold
test_type = 'smc'
use_value = False
resultsfile = './results/forward_sample_alpha.p' #alpha means first try on saturday
debug = False
use_prev_value = False

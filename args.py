
pooling = 'mean'

h_out = 128

import string
num_char_types = len(string.printable[:-4]) + 1

char_embed_dim = 20

column_encoding_dim = 16

strLen = 100

encoder = 'dense'

num_dense_layers = 10

growth_rate = 56

button_embed_dim = 32

batchsize = 2000

train_iterations = 100000

save_path = 'robut_model.p'

print_freq = 20

save_freq = 100

test_freq = 1000
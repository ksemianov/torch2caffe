require 'nn'
require 'xlua'
require 'json'
require 'paths'

npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')

input = npy4th.loadnpy(arg[2]):float()

model = torch.load(arg[1])

model:evaluate()

out = model:forward(input)
npy4th.savenpy('data/out_torch.npy', out)

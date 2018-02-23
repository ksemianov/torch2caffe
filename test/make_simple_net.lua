require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(3, 10, 2, 2, 1, 1, 1, 1))
cnn:add(nn.ReLU())
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
cnn:add(nn.SpatialConvolution(10, 10, 5, 5, 1, 1, 2, 2))
cnn:add(nn.SpatialBatchNormalization(10))
cnn:add(nn.ELU())

torch.save('models/simple_net.th', cnn)

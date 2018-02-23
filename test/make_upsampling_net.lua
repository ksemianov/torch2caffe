require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(3, 10, 2, 2))
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
cnn:add(nn.SpatialUpSamplingNearest(2))
cnn:add(nn.SpatialConvolution(10, 10, 2, 2))
cnn:add(nn.SpatialBatchNormalization(10))

cnn:add(nn.SpatialAveragePooling(2, 2, 2, 2, 1, 1))
cnn:add(nn.SpatialUpSamplingBilinear(2))
cnn:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
cnn:add(nn.SpatialConvolution(10, 10, 2, 2))
cnn:add(nn.SpatialBatchNormalization(10))

torch.save('models/upsampling_net.th', cnn)

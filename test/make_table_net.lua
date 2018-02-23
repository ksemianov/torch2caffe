require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

local cnn = nn.Sequential()
cnn:add(nn.SpatialConvolution(3, 2, 2, 2, 1, 1, 1, 1))
cnn:add(nn.ELU())
--cnn:add(nn.Parallel(1, 1)
    --:add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1))
    --:add(nn.SpatialAveragePooling(3, 3, 2, 2, 1, 1)))
cnn:add(nn.SpatialConvolution(2, 5, 2, 2, 1, 1, 1, 1))
cnn:add(nn.SpatialBatchNormalization(5))
cnn:add(nn.ReLU())
cnn:add(nn.ConcatTable()
    :add(nn.Sequential():add(nn.SpatialConvolution(5, 10, 2, 2, 1, 1, 1, 1)))
    :add(nn.Sequential():add(nn.SpatialConvolution(5, 10, 2, 2, 1, 1, 1, 1))))
cnn:add(nn.CAddTable())
cnn:add(nn.SpatialFullConvolution(10, 3, 2, 2))
cnn:add(nn.ConcatTable()
    :add(nn.Sequential():add(nn.SpatialMaxPooling(2, 2, 2, 2, 1, 1)))
    :add(nn.Sequential():add(nn.SpatialAveragePooling(2, 2, 2, 2, 1, 1))))
cnn:add(nn.CAddTable())

torch.save('models/table_net.th', cnn)

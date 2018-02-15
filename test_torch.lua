require 'nn'
require 'xlua'
require 'json'
require 'paths'

npy4th = require 'npy4th'

local absorb_bn_conv = function (w, b, mean, invstd, affine, gamma, beta)
  w:cmul(invstd:view(w:size(1),1):repeatTensor(1,w:nElement()/w:size(1)))
  b:add(-mean):cmul(invstd)

  if affine then
    w:cmul(gamma:view(w:size(1),1):repeatTensor(1,w:nElement()/w:size(1)))
    b:cmul(gamma):add(beta)
  end
end


local absorb_bn_deconv = function (w, b, mean, invstd, affine, gamma, beta)
   w:cmul(invstd:view(b:size(1),1):repeatTensor(w:size(1),w:nElement()/w:size(1)/b:nElement()))
   b:add(-mean):cmul(invstd)

   if affine then
      w:cmul(gamma:view(b:size(1),1):repeatTensor(w:size(1),w:nElement()/w:size(1)/b:nElement()))
      b:cmul(gamma):add(beta)
   end
end

local backward_compat_running_std = function(x, i)
   if x.modules[i].running_std then
      x.modules[i].running_var = x.modules[i].running_std:pow(-2):add(-x.modules[i].eps)
      x.modules[i].running_std = nil
   end
end

local function batchnorm_absorber(x)
   local i = 1
   while (i <= #x.modules) do
      if x.modules[i].__typename == 'nn.Sequential' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.Parallel' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.Concat' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.DepthConcat' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.DataParallel' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.ModelParallel' then
         batchnorm_absorber(x.modules[i])
      elseif x.modules[i].__typename == 'nn.ConcatTable' then
         batchnorm_absorber(x.modules[i])
      else
         -- check BN
         if x.modules[i].__typename == 'nn.SpatialBatchNormalization' then
            backward_compat_running_std(x, i)
            if x.modules[i-1] and
              (x.modules[i-1].__typename == 'nn.SpatialConvolution' or
               x.modules[i-1].__typename == 'nn.SpatialConvolutionMM') then
               if x.modules[i-1].bias == nil then
                 local empty_bias = torch.Tensor() --we set default tensor type
                 empty_bias:resizeAs(x.modules[i].running_mean)
                 empty_bias:fill(0)
                 x.modules[i-1].bias = empty_bias
               end
               absorb_bn_conv(x.modules[i-1].weight,
                              x.modules[i-1].bias,
                              x.modules[i].running_mean,
                              x.modules[i].running_var:clone():add(x.modules[i].eps):pow(-0.5),
                              x.modules[i].affine,
                              x.modules[i].weight,
                              x.modules[i].bias)
               x:remove(i)
               i = i - 1
            elseif x.modules[i-1] and
                  (x.modules[i-1].__typename == 'nn.SpatialFullConvolution') then
               absorb_bn_deconv(x.modules[i-1].weight,
                                x.modules[i-1].bias,
                                x.modules[i].running_mean,
                                x.modules[i].running_var:clone():add(x.modules[i].eps):pow(-0.5),
                                x.modules[i].affine,
                                x.modules[i].weight,
                                x.modules[i].bias)
               x:remove(i)
               i = i - 1
            else
               print('Skipping bn absorb: Convolution module must exist right before batch normalization layer')
            end
         elseif x.modules[i].__typename == 'nn.BatchNormalization' then
            backward_compat_running_std(x, i)
            if x.modules[i-1] and
              (x.modules[i-1].__typename == 'nn.Linear') then
               absorb_bn_conv(x.modules[i-1].weight,
                              x.modules[i-1].bias,
                              x.modules[i].running_mean,
                              x.modules[i].running_var:clone():add(x.modules[i].eps):pow(-0.5),
                              x.modules[i].affine,
                              x.modules[i].weight,
                              x.modules[i].bias)
               x:remove(i)
               i = i - 1
            else
               assert(false, 'Convolution module must exist right before batch normalization layer')
            end
         end
      end
      i = i + 1
   end

   collectgarbage()
   return x
end

local function optimize(net)
  return batchnorm_absorber(net)
end



input = npy4th.loadnpy('x.npy')

model = torch.load("genModel1")

model:evaluate()

out = model:forward(input)
print(model)
print(out[1][1][1][1])
print(out[1][1][2][1])
print(out[1][2][1][1])
print(out[1][2][2][1])
print(out[1][3][1][1])


model = optimize(model)
print(model)
out = model:forward(input)

print(out[1][1][1][1])
print(out[1][1][2][1])
print(out[1][2][1][1])
print(out[1][2][2][1])
print(out[1][3][1][1])

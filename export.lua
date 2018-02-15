------------------------------------------------------------------
-- Export torch model config and layer param to disk.
------------------------------------------------------------------

require 'nn'
require 'xlua'
require 'json'
require 'paths'

npy4th = require 'npy4th';

torch.setdefaulttensortype('torch.FloatTensor')


PARAM_DIR = './param/'    -- Directory for saving layer param.
CONFIG_DIR = './config/'  -- Directory for saving net config.


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

---------------------------------------------------------------
-- Save layer param to disk.
-- Note we convert default FloatTensor to DoubleTensor
-- because of a weird bug in `npy4th`...
--
function save_param(save_name, weight, bias)
    npy4th.savenpy(PARAM_DIR..save_name..'.w.npy', weight:double())
    if bias then npy4th.savenpy(PARAM_DIR..save_name..'.b.npy', bias:double()) end
end

---------------------------------------------------------------
-- Save conv layer.
--
function conv_layer(layer, current, prev)
    local layer_name = current
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.nOutputPlane
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Convolution',
        ['name'] = layer_name,
        ['num_output'] = nOutput,
        ['kW'] = kW,
        ['kH'] = kH,
        ['dW'] = dW,
        ['dH'] = dH,
        ['pW'] = pW,
        ['pH'] = pH,
        ['prev'] = prev,
    }
end

---------------------------------------------------------------
-- Save deconv layer.
--
function deconv_layer(layer, current, prev)
    local layer_name = current
    save_param(layer_name, layer.weight, layer.bias)

    local nOutput = layer.nOutputPlane
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Deconvolution',
        ['name'] = layer_name,
        ['num_output'] = nOutput,
        ['kW'] = kW,
        ['kH'] = kH,
        ['dW'] = dW,
        ['dH'] = dH,
        ['pW'] = pW,
        ['pH'] = pH,
	    ['prev'] = prev,
    }
end


---------------------------------------------------------------
-- Save pooling layer.
--
function pooling_layer(layer, current, prev)
    local layer_name = current

    local pool_type = torch.type(layer)=='nn.SpatialMaxPooling' and 0 or 1
    local kW,kH = layer.kW,layer.kH
    local dW,dH = layer.dW,layer.dH
    local pW,pH = layer.padW,layer.padH

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Pooling',
        ['name'] = layer_name,
        ['pool_type'] = pool_type,
        ['kW'] = kW,
        ['kH'] = kH,
        ['dW'] = dW,
        ['dH'] = dH,
        ['pW'] = pW,
        ['pH'] = pH,
        ['prev'] = prev,
    }
end


---------------------------------------------------------------
-- Save bn runing_mean & running_var, and split weight & bias.
-- The reason for doing this is caffe uses BN+Scale to achieve
-- the full torch BN functionality.
--
function bn_layer(layer, current, prev)
    -- Save running_mean & running_var.
    local layer_name = current
    save_param(layer_name, layer.running_mean, layer.running_var)

    local affine = layer.weight and true or false
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'BatchNorm',
        ['name'] = layer_name,
        ['affine'] = affine,
	['prev'] = prev,
    }

    -- If affine=true, save BN weight & bias as Scale layer param.
    if affine then
    layer_name1 =  tostring(tonumber(layer_name)+1)
    save_param(layer_name1, layer.weight, layer.bias)

    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Scale',
        ['name'] = layer_name1,
	['prev'] = layer_name,
    }
    end
end

---------------------------------------------------------------
-- Save concat layer.
--
function concat_layer(layer, current, prev)
    local layer_name = current
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Concat',
        ['name'] = layer_name,
	    ['prev'] = prev,
    }
end

---------------------------------------------------------------
-- Save tableconcat + CAddTable layer. We use only one layer in caffe elmwise instead 2 in torch
--
function cadd_layer(layer, current, prev)
    local layer_name = current
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'Cadd',
        ['name'] = layer_name,
        ['prev'] = prev,
    }
end


---------------------------------------------------------------
-- Save relu layer.
--
function relu_layer(layer, current, prev)
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'ReLU',
        ['name'] = "relu" .. current,
	    ['prev'] = prev,
    }
end

---------------------------------------------------------------
-- Save elu layer.
--
function elu_layer(layer, current, prev)
    net_config[#net_config+1] = {
        ['id'] = #net_config,
        ['type'] = 'ELU',
        ['name'] = "elu" .. current,
        ['prev'] = prev,
	    ['alpha'] = layer.alpha,
    }
end

if #arg ~= 5 then
    print('Usage: th export.lua [path_to_torch_model] [input_shape]')
    print('e.g. th export.lua ./net.t7 {1,1,28,28}')
    return
else
    net_path = arg[1]
    input_shape = { tonumber(arg[2]), tonumber(arg[3]),
                    tonumber(arg[4]), tonumber(arg[5]) }
end

paths.mkdir(PARAM_DIR)
paths.mkdir(CONFIG_DIR)

net = torch.load(net_path)
net:evaluate()
net_config = {}
net = optimize(net)


-- Add input layer config.
net_config[#net_config+1] = {
    ['id'] = #net_config,
    ['type'] = 'DummyData',
    ['name'] = '0',
    ['input_shape'] = input_shape,
}

-- Map layer type to it's saving function.
layerfn = {
    ['nn.SpatialConvolution'] = conv_layer,
    ['nn.SpatialBatchNormalization'] = bn_layer,
    ['nn.SpatialMaxPooling'] = pooling_layer,
    ['nn.SpatialAveragePooling'] = pooling_layer,
    ['nn.Concat'] = concat_layer,
    ['nn.ReLU'] = relu_layer,
    ['nn.ELU'] = elu_layer,
    ['nn.SpatialFullConvolution'] = deconv_layer,
    ['nn.ConcatTable'] = cadd_layer,
}





mod = {}
current = 1

function getDescr(model, mod, current, prev)
    isUsed = 1
    for i=1,#model do
        local layer = model:get(i)
        layer_type = torch.type(layer)
        save = layerfn[layer_type]
        if (layer_type == "nn.Concat" or layer_type == "nn.ConcatTable") then
            local layer1 = layer
            local save1= save
            inp = current-1
            strr = ""
            local _prev = current
            for i=1,#layer.modules do
                if torch.type(layer.modules[i]) == "nn.Identity" then
		    strr = strr .. tostring(inp).. ","
                else
                    current = getDescr(layer.modules[i], mod, current, inp)
                    if _prev == current then
                        strr = strr .. tostring(inp).. ","
                    else
                       strr = strr .. tostring(current-1).. ","
                       _prev = current
                    end
                end
            end
            table.insert(mod,tostring("concat" .. tostring(current) .. " " ..strr))
            save1(layer1, tostring(current), strr)
            current = current + 1
         elseif (layer_type == "nn.Sequential" ) then
              print(current)
              current = getDescr(layer, mod, current, current)
         elseif (layer_type == "nn.CAddTable" or layer_type == "nn.Identity") then
            --pass layer, already make it in concatTable
            i = i
         elseif (layer_type == "nn.ReLU" or layer_type == "nn.ELU") then
            save(layer, tostring(current-1), tostring(current-1))
            table.insert(mod,tostring(tostring(current-1) .. " " ..  tostring(current -1)))
         else
            if isUsed==0 then
                save(layer, tostring(current), tostring(current-1))
                table.insert(mod,tostring(tostring(current) .. " " ..  tostring(current -1)))
                current = current + 1
            else
                save(layer, tostring(current), tostring(prev))
                table.insert(mod,tostring(tostring(current) .. " " ..  tostring(prev)))
                isUsed = 0
                current = current + 1
	    end
	    if (layer_type == "nn.SpatialBatchNormalization") then
	        table.insert(mod,tostring("batch" .. tostring(current) .."and".. tostring(current-1) .. " " ..  tostring(current -2)))
            	current = current + 1
	    end
        end
    end
    return current
end

getDescr(net, mod, current, 0)
print(mod)



-- Save config file.
json.save(CONFIG_DIR..'net.json', net_config)

-- Graph.
graph = torch.zeros(#net_config, #net_config)

-- TODO: build graph from net structure.
-- For now just sequential.
for i = 1, graph:size(1) do
    graph[i][i] = 1
    if i < graph:size(1) then
        graph[i][i+1] = 1
    end
end
npy4th.savenpy(CONFIG_DIR..'graph.npy', graph)


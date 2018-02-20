'''Import layer params from disk to rebuild the caffemodel.'''

from __future__ import print_function

import os
os.environ['GLOG_minloglevel'] = '2'  # Hide caffe debug info.

import math
import json
import caffe
import numpy as np

from caffe import layers as L, params as P, to_proto

# Directory containing layer param and config file.
PARAM_DIR = './param/'
CONFIG_DIR = './config/'


#Create input layer in caffe
def input_layer(layer_config):
    input_shape = layer_config['input_shape']
    return L.DummyData(shape=[dict(dim=input_shape)], ntop=1)


#Create conv layer.
def conv_layer(layer_config, bottom_name):
    num_output = layer_config['num_output']
    kW, kH = layer_config['kW'], layer_config['kH']
    dW, dH = layer_config['dW'], layer_config['dH']
    pW, pH = layer_config['pW'], layer_config['pH']
    return L.Convolution(name = layer_config['name'],
                         ntop = 0, top = layer_config['name'], #use ntop=0, for set top manually
                         num_output=num_output,
                         bottom=bottom_name,
                         kernel_w=kW,
                         kernel_h=kH,
                         stride_w=dW,
                         stride_h=dH,
                         pad_w=pW,
                         pad_h=pH)


#Create deconv layer.
def deconv_layer(layer_config, bottom_name):
    num_output = layer_config['num_output']
    kW, kH = layer_config['kW'], layer_config['kH']
    dW, dH = layer_config['dW'], layer_config['dH']
    pW, pH = layer_config['pW'], layer_config['pH']
    adj = layer_config['adj']
    if adj>0:
	kW = kW + 1
	kH = kH + 1
    return L.Deconvolution(name = layer_config['name'],
                         bottom=bottom_name,
                         ntop = 0, top = layer_config['name'],
                         convolution_param = dict(kernel_w=kW,
						  kernel_h=kH,
                         			  num_output=num_output,
                         			  stride_w=dW,
                         			  stride_h=dH,
                         			  pad_w=pW,
                         			  pad_h=pH))

#Create upsample layer. Support only x2 upsample
def upsample_layer(layer_config, bottom_name):
    num_output1 = layer_config['num_output']
    scale = layer_config['scale']
    if scale != 2:
	raise ValueError('Scale factor should be 2')
    return L.Deconvolution(name = layer_config['nameInner'],
                         bottom=bottom_name,
                         ntop = 0, top = layer_config['name'],
                         convolution_param = dict(kernel_w= scale * scale - scale % 2,
                                                  kernel_h= scale * scale - scale % 2,
                                                  num_output = num_output1,
						  group = num_output1,
                                                  stride_w=scale,
                                                  stride_h=scale,
                                                  pad_w=int(math.ceil((scale - 1) / 2.)), 
                                                  pad_h=int(math.ceil((scale - 1) / 2.)),
			 			  weight_filler=dict(type='bilinear'),
			 			  bias_term=False),
			param = dict(lr_mult=0, decay_mult= 0))


#Create bn layer.
def bn_layer(layer_config, bottom_name):
    return L.BatchNorm(ntop = 0, top = layer_config['name'],
		       bottom=bottom_name,
                       use_global_stats=True)

def scale_layer(layer_config, bottom_name):
    return L.Scale(ntop = 0, top = layer_config['name'],
		   bottom=bottom_name,
                   bias_term=True)

#Create concat layer.
def concat_layer(layer_config, bottom_name):
    bottom = bottom_name.split(',')
    bottom.pop()
    return L.Concat(ntop = 0, top = layer_config['name'],
                    name = layer_config['name'],
                    bottom=bottom)


#Create Eltwise layer.
def cadd_layer(layer_config, bottom_name):
    '''For ReLU layer, top=bottom'''
    bottom = bottom_name.split(',') 
    bottom.pop()
    return L.Eltwise(name = layer_config['name'],
                     ntop = 0, top = layer_config['name'],
                     bottom=bottom,
                     operation=P.Eltwise.SUM)


#Create relu layer.
def relu_layer(layer_config, bottom_name):
    '''For ReLU layer, top=bottom(caffe feature)'''
    return L.ReLU(name = layer_config['name'],
                  bottom=bottom_name,
                  top=bottom_name,
                  in_place=True)

#Create elu layer.
def elu_layer(layer_config, bottom_name):
    '''For ELU layer, top=bottom(caffe feature)'''
    return L.ELU(name = layer_config['name'],
                 bottom=bottom_name,
                 top=bottom_name,
                 alpha= layer_config['alpha'],
                 in_place=True)


#Create pool(max, average) layer.
def pool_layer(layer_config, bottom_name):
    pool_type = layer_config['pool_type']
    kW, kH = layer_config['kW'], layer_config['kH']
    dW, dH = layer_config['dW'], layer_config['dH']
    pW, pH = layer_config['pW'], layer_config['pH']
    if pW !=1 or pH !=1:
        raise ValueError('padding into pooling should be 1')
    return L.Pooling(name = layer_config['name'],
                     ntop = 0, top = layer_config['name'],
                     bottom=bottom_name,
                     pool=pool_type,
                     kernel_w=kW,
                     kernel_h=kH,
                     stride_w=dW,
                     stride_h=dH,
                     pad_w=0,
                     pad_h=0)


def build_prototxt():
    '''Build a new prototxt from config file.

    Save as `cvt_net.prototxt`.
    '''
    print('==> Building prototxt..')

    # Map layer_type to its building function.
    layer_fn = {
        'Data': input_layer,
        'DummyData': input_layer,
        'Convolution': conv_layer,
        'Deconvolution': deconv_layer,
        'BatchNorm': bn_layer,
        'Scale': scale_layer,
        'ReLU': relu_layer,
        'ELU': elu_layer,
        'Pooling': pool_layer,
        'Concat' : concat_layer,
        'Cadd' : cadd_layer,
	'Upsample' :upsample_layer,
    }

    net = caffe.NetSpec()

    with open(CONFIG_DIR + 'net.json', 'r') as f:
        net_config = json.load(f)

    # DFS graph to build prototxt.
    graph = np.load(CONFIG_DIR + 'graph.npy')
    num_nodes = graph.shape[0]

    def dfs():
	print('... Add layer: DummyData')
        input_layer_name = net_config[0]['name']
        net[input_layer_name] = input_layer(net_config[0])    
        for w in range(1, num_nodes):
	    print(net_config[w])
            bottom_layer_name = net_config[w]["prev"]
       	    layer_config = net_config[w]
            layer_name = str(layer_config['name'])
            layer_type = layer_config['type']

            print('... Add layer: %s %s' % (layer_type, layer_name))
            get_layer = layer_fn.get(layer_type)
            if not get_layer:
                raise TypeError('%s not supported yet!' % layer_type)

            layer = get_layer(layer_config, bottom_layer_name)
            net[layer_name] = layer
	    print(layer_name)
    # DFS.
    dfs()

    # Save prototxt.
    with open('cvt_net.prototxt', 'w') as f:
        f.write(str(net.to_proto()))
        print('Saved!\n')

def load_param(layer_name):
    '''Load saved layer params.

    Returns:
      (tensor) weight or running_mean or None.
      (tensor) bias or running_var or None.
    '''
    weight_path = PARAM_DIR + layer_name + '.w.npy'
    bias_path = PARAM_DIR + layer_name + '.b.npy'

    weight = np.load(weight_path) if os.path.isfile(weight_path) else None
    bias = np.load(bias_path) if os.path.isfile(bias_path) else None

    return weight, bias

def fill_params():
    '''Fill network with saved params.

    Save as `cvt_net.caffemodel`.
    '''
    print('==> Filling layer params..')
    
    #matrix for computing x2 upsample
    w = [[0.25, 0.5,  0.25, 0], [0.5,  1.,   0.5 , 0], [0.25, 0.5,  0.25, 0], [0,0,0,0]]
    
    net = caffe.Net('cvt_net.prototxt', caffe.TEST)
    for i in range(len(net.layers)):
        layer_name = net._layer_names[i]
        layer_type = net.layers[i].type

        print('... Layer %d : name: %s type: %s' % (i, layer_name, layer_type))
	
        weight, bias = load_param(layer_name)
	if layer_type == "Deconvolution" and layer_name[0:2] =="Up":
        #if deconv with adjW, adjH, need to move weight
	    print("in here")
	    net.params[layer_name][0].data[0] =w
            net.params[layer_name][0].data[1] =w
            net.params[layer_name][0].data[2] =w

	#if upsample, we don't reload weight
        else:
            if (weight is not None) and layer_type != "Eltwise":
                print(weight.shape)
		net.params[layer_name][0].data[...] = weight
            if (bias is not None) and layer_type != "Eltwise":
                net.params[layer_name][1].data[...] = bias

            if layer_type == 'BatchNorm':
                net.params[layer_name][2].data[...] = 1.  # use_global_stats=true

    net.save('cvt_net.caffemodel')
    print('Saved!')


if __name__ == '__main__':
    # Build new prototxt based on config file.
    build_prototxt()

    # Fill network with saved params.
    fill_params()




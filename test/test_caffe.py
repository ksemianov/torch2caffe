import os
import sys
os.environ['GLOG_minloglevel'] = '2' # hide debug log

import caffe
import numpy as np

prototxt = './cvt_net.prototxt'
binary = './cvt_net.caffemodel'

caffe.set_mode_cpu()
net = caffe.Net(prototxt, binary, caffe.TEST)

x = np.load(sys.argv[1])
net.blobs['0'].data[...] = x
out = net.forward()[list(net.blobs)[-1]]
out = np.array(out)
np.save("data/out_caffe.npy", out)

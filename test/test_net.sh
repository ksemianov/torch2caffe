#!/bin/sh

set -e

th test/make_$1.lua
th export.lua models/$1.th {10,3,128,128}
python import.py 

python test/generate_x.py 10 3 128 128 data/input.npy
th test/test_torch.lua models/$1.th data/input.npy
python test/test_caffe.py data/input.npy

python test/compare.py data/out_torch.npy data/out_caffe.npy

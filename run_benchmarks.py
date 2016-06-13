"""
feed it a model name, will run benchmarks for that model
"""
import sys
import time
import importlib
import numpy as np
import argparse
#from mycltensor import MyClTensor
#from neon.layers.layer import Convolution
#from neon.backends.make_backend import make_backend

parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='cl')
parser.add_argument('--model', default='vgga')
args = parser.parse_args()

model_name = args.model
backend_name = args.backend
print('model_name', model_name, 'backend_name', backend_name)

its = 10

model = importlib.import_module('models.%s' % model_name)
backend = importlib.import_module('backends.%s' % backend_name)

def test(backend, batch_size, its, layer):
    backend_obj = backend.Test(batch_size, its, layer)
    fpropCumTime = 0
    bpropCumTime = 0
    for it in range(its):
        backend_obj.sync()
        start = time.time()

        backend_obj.fprop()

        backend_obj.sync()
        end = time.time()

        fprop = end - start
        fpropCumTime += end - start

        backend_obj.sync()
        start = time.time()

        backend_obj.bprop()

        backend_obj.sync()
        end = time.time()

        bprop = end - start
        bpropCumTime += end - start
        print('fprop %.3f bprop %.3f' % (fprop, bprop))
    print('avg fprop %.3f bprop %.3f' % (fpropCumTime / its, bpropCumTime / its))

batch_size = model.get_batchsize()
print('batch_size', batch_size)
for layer in model.get_net():
    if layer['Ci'] >= 4:
        print('RUNNING', layer)
        test(backend, batch_size, its, layer)
    else:
        print('SKIPPING', layer)


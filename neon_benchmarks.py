"""
feed it a model name, will run benchmarks for that model
"""
import sys
import time
import importlib
import numpy as np
import argparse
import cpuref
import random

def check_outputs(batch_size, layer_def, gpu_O, I, W):
    input_filters = layer_def['Ci']
    output_filters = layer_def['Co']
    image_size = layer_def['iW']
    kernel_size = layer_def['kH']

    random.seed(123)
    cpuref.check_O(gpu_O=gpu_O, W=W, I=I, c=0, h=0, w=0, n=0)
    for i in range(10):  # draw 10 samples
        co = random.randint(0, output_filters - 1)
        oh = random.randint(0, image_size - 1)
        ow = random.randint(0, image_size - 1)
        n = random.randint(0, batch_size - 1)
        cpuref.check_O(gpu_O=gpu_O, W=W, I=I, c=co, h=oh, w=ow, n=n)

def test(backend, batch_size, its, layer_def):
    assert layer_def['iH'] == layer_def['iW']
    assert layer_def['kH'] == layer_def['kW']

    input_filters = layer_def['Ci']
    output_filters = layer_def['Co']
    image_size = layer_def['iW']
    kernel_size = layer_def['kH']

    np.random.seed(123)

    I = np.zeros((input_filters,image_size, image_size, batch_size), dtype=np.float32)
    I[:] = np.random.randn(*I.shape)
    W = np.random.randn(input_filters, kernel_size, kernel_size, output_filters).astype(np.float32)
    gradO = np.random.randn(image_size * image_size * output_filters, batch_size).astype(np.float32)

    backend_obj = backend.Test(batch_size=batch_size, its=its, layer_def=layer_def, I=I, W=W, gradO=gradO)

    # check correctness, for a few values, (this also serves as warmup):
    backend_obj.fprop()
    O = backend_obj.getO()
    check_outputs(batch_size, layer_def, gpu_O=O, I=I, W=W)

    backend_obj.bprop()
    gradW = backend_obj.getGradW()
    gradI = backend_obj.getGradI()
    
    backend_obj.sync()

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
    return (fpropCumTime / its, bpropCumTime / its)

if __name__ == '__main__':
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

    batch_size = model.get_batchsize()
    print('batch_size', batch_size)
    results = []
    for i, layer_def in enumerate(model.get_net()):
        if layer_def['Ci'] >= 4:
            print('RUNNING', layer_def)
            res = test(backend, batch_size, its, layer_def)
#            results.append(res)
            results.append('Layer %s: fprop=%.3f bprop=%.3f' % (i, res[0], res[1]))
        else:
            print('SKIPPING', layer_def)
            results.append('Layer %s: SKIPPING' % i)
    print('')
    print('Results')
    print('-------')
    for res in results:
        print(res)
    print('')


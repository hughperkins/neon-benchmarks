"""
feed it a model name, will run benchmarks for that model
"""
from __future__ import print_function, division
import sys
import time
import logging
import importlib
import numpy as np
import argparse
import cpuref
import random
import traceback

logger = logging.getLogger()

def check_outputs(batch_size, layer_def, gpu_O, I, W, eps=1e-4):
    input_filters = layer_def['Ci']
    output_filters = layer_def['Co']
    image_size = layer_def['iW']
    kernel_size = layer_def['kH']

    random.seed(123)
    # prepare params now, in case something else modifies seed later
    params = []
    params.append({'c': 0, 'h': 0, 'w': 0, 'n': 0})
    for i in range(10):  # draw 10 samples
        co = random.randint(0, output_filters - 1)
        oh = random.randint(0, image_size - 1)
        ow = random.randint(0, image_size - 1)
        n = random.randint(0, batch_size - 1)
        params.append({'c': co, 'h': oh, 'w': ow, 'n': n})
#    cpuref.check_O(gpu_O=gpu_O, W=W, I=I, c=0, h=0, w=0, n=0, eps=eps)
    diffs = []
    for param in params:
        diffs.append(cpuref.check_O(gpu_O=gpu_O, W=W, I=I, eps=eps, **param))
    return sum(diffs) / len(diffs)

def check_gradW(batch_size, layer_def, gpu_gradW, I, W, gradO, eps=1e-4):
    input_filters = layer_def['Ci']
    output_filters = layer_def['Co']
    image_size = layer_def['iW']
    kernel_size = layer_def['kH']

    diffs = []
    random.seed(123)
    gpu_gradW = gpu_gradW.reshape((input_filters, kernel_size, kernel_size, output_filters))
    diff = cpuref.check_gradW(gradW=gpu_gradW, W=W, I=I, gradO=gradO, ci=0, h=0, w=0, co=0, eps=eps)
    diffs.append(diff)
    for i in range(10):  # draw 10 samples
        co = random.randint(0, output_filters - 1)
        kh = random.randint(0, kernel_size - 1)
        kw = random.randint(0, kernel_size - 1)
        ci = random.randint(0, input_filters - 1)
        diff = cpuref.check_gradW(gradW=gpu_gradW, W=W, I=I, gradO=gradO, co=co, h=kh, w=kw, ci=ci, eps=eps)
        diffs.append(diff)
    return sum(diffs) / len(diffs)

def check_gradI(batch_size, layer_def, gpu_gradI, W, gradO, eps=1e-4):
    input_filters = layer_def['Ci']
    output_filters = layer_def['Co']
    image_size = layer_def['iW']
    kernel_size = layer_def['kH']

    diffs = []
    random.seed(123)
    gpu_gradI = gpu_gradI.reshape((input_filters, image_size, image_size, batch_size))
    diffs.append(cpuref.check_gradI(gradI=gpu_gradI, W=W, gradO=gradO, c=0, h=0, w=0, n=0, eps=eps))
    for i in range(10):  # draw 10 samples
        ci = random.randint(0, input_filters - 1)
        ih = random.randint(0, image_size - 1)
        iw = random.randint(0, image_size - 1)
        n = random.randint(0, batch_size - 1)
        diffs.append(cpuref.check_gradI(gradI=gpu_gradI, W=W, gradO=gradO, c=ci, h=ih, w=iw, n=n, eps=eps))
    return sum(diffs) / len(diffs)

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

    # we copy the tensors, to make sure the backend doesnt sneakily modify them itself :-P
    backend_obj = backend.Test(batch_size=batch_size, its=its, layer_def=layer_def, I=np.copy(I), W=np.copy(W), gradO=np.copy(gradO))

    # check correctness, for a few values, (this also serves as warmup):
    backend_obj.fprop()
    O = backend_obj.getO()
    O_diffs = check_outputs(batch_size, layer_def, gpu_O=O, I=I, W=W)

    backend_obj.bprop()
    gradW = backend_obj.getGradW()
    gradI = backend_obj.getGradI()
    gradW_diffs = check_gradW(batch_size, layer_def, gpu_gradW=gradW, I=I, W=W, gradO=gradO)
    gradI_diffs = check_gradI(batch_size, layer_def, gpu_gradI=gradI, W=W, gradO=gradO)
    
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
    return {
        'fprop': fpropCumTime / its, 'bprop': bpropCumTime / its, 'eps_O': O_diffs,
        'eps_gradW': gradW_diffs, 'eps_gradI': gradI_diffs}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', default='winogradcl')
    parser.add_argument('--model', default='vgga')
    parser.add_argument('--layer', default='all', help='zero-indexed')
    parser.add_argument('--loglevel', default='INFO')
    args = parser.parse_args()

    model_name = args.model
    backend_name = args.backend
    logging.basicConfig()
    logger.setLevel(args.loglevel.upper())
    print('model_name', model_name, 'backend_name', backend_name)

    its = 10

    model = importlib.import_module('models.%s' % model_name)
    backend = importlib.import_module('backends.%s' % backend_name)

    batch_size = model.get_batchsize()
    print('batch_size', batch_size)
    results = []
    for i, layer_def in enumerate(model.get_net()):
        if args.layer != 'all' and str(i) != args.layer:
            continue
        try:
            print(layer_def)
            res = test(backend, batch_size, its, layer_def)
#            results.append(res)
            results.append('Layer %s: fprop=%.3f bprop=%.3f eps_O=%.0e eps_gradW=%.0e eps_gradI=%.0e' % (
                i, res['fprop'], res['bprop'], res['eps_O'], res['eps_gradW'], res['eps_gradI']))
        except Exception as e:
            logger.debug(traceback.format_exc())
            print('.. SKIPPED')
            results.append('Layer %s: SKIPPED' % i)
    print('')
    print('Results')
    print('-------')
    for res in results:
        print(res)
    print('')


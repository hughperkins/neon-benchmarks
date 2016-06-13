"""
Tests using neonbase, ie assumes we are on nervanasystems neon master branch
"""
import sys
import time
import numpy as np
from neon.layers import Convolution
from neon.initializers import Gaussian
from neon.backends import gen_backend
import pycuda.driver as cuda
#import pycuda.autoinit
import pycuda.gpuarray as gpuarray

init = Gaussian()

def test(batch_size, its, layer):
    np.random.seed(123)

    gen_backend(backend='gpu', batch_size=batch_size,
            datatype=np.float32, device_id=0)

    input_filters = layer['Ci']
    output_filters = layer['Co']
    image_size = layer['iW']
    assert layer['iH'] == image_size
    assert layer['kH'] == layer['kW'] == 3

    inputs = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
    inputs[:] = np.random.randn(*inputs.shape)
    inputs_cuda = gpuarray.to_gpu(inputs)

    gradOutputs = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
    gradOutputs[:] = np.random.randn(*gradOutputs.shape)
    gradOutputs_cuda = gpuarray.to_gpu(gradOutputs)

    conv = Convolution((3, 3, output_filters), strides=1, padding=1, init=init)
    conv.configure((input_filters, image_size, image_size))
    conv.allocate()
    
    conv.fprop(inputs_cuda)
    conv.bprop(gradOutputs_cuda)


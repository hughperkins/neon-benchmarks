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

class Test(object):
    def __init__(self, batch_size, its, layer_def, W, I, gradO):
        gen_backend(backend='gpu', batch_size=batch_size,
                datatype=np.float32, device_id=0)

        assert layer_def['iH'] == layer_def['iW']
        assert layer_def['kH'] == layer_def['kW']
        assert layer_def['dH'] == layer_def['dW']
        assert layer_def['padH'] == layer_def['padW']

        input_filters = layer_def['Ci']
        output_filters = layer_def['Co']
        image_size = layer_def['iW']
        filter_size = layer_def['kH']
        padding = layer_def['padH']
        stride = layer_def['dH']

        self.I = I
        self.W = W
        self.gradO = gradO

        I_cuda = gpuarray.to_gpu(I)
        gradO_cuda = gpuarray.to_gpu(gradO)
        W_cuda = gpuarray.to_gpu(W)

        conv = Convolution((filter_size, filter_size, output_filters), strides=stride, padding=padding, init=init)
        conv.configure((input_filters, image_size, image_size))
        conv.allocate()
#        conv.allocate_deltas()
        conv.W = W_cuda

        self.conv = conv
        deltas = np.zeros(I.shape, dtype=np.float32)
        deltas_cuda = gpuarray.to_gpu(deltas)
        conv.deltas = deltas_cuda

#        self.O = O
#        self.gradW = gradW
#        self.gradI = gradI

        self.I_cuda = I_cuda
        self.O_cuda = conv.outputs
        self.gradO_cuda = gradO_cuda

        self.gradW_cuda = conv.dW
        self.gradI_cuda = conv.deltas

    def sync(self):
        cuda.Context.synchronize()

    def fprop(self):
        self.conv.fprop(self.I_cuda)

    def bprop(self):
        res = self.conv.bprop(self.gradO_cuda)

    def getO(self):
        return self.O_cuda.get()

    def getGradW(self):
        return self.gradW_cuda.get()

    def getGradI(self):
        return self.conv.deltas.get()


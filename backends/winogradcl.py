# tests using opencl backend, in winogradcl branch

import sys
import time
import importlib
import numpy as np
from winogradcl.backends.kernels.cl.mycltensor import MyClTensor
from winogradcl.layers.layer import Convolution

from winogradcl.backends.make_backend import make_backend

class Test(object):
    def __init__(self, batch_size, its, layer_def, W, I, gradO):
        input_filters = layer_def['Ci']
        output_filters = layer_def['Co']
        image_size = layer_def['iW']
        assert layer_def['iH'] == image_size
        assert layer_def['kH'] == layer_def['kW'] == 3

        self.W = W
        self.I = I
        self.gradO = gradO

        be = make_backend(batch_size=batch_size,
                datatype=np.float32, device_id=0).be

        W_cl = MyClTensor.from_np(be, self.W)
#        print('self.I', self.I)
        I_cl = MyClTensor.from_np(be, self.I)

        conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        conv.configure((input_filters,image_size, image_size))
        conv.W = W_cl

        O = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        O_cl = MyClTensor.from_np(be, O)
        conv.outputs = O_cl

        conv.fprop(I_cl)
        be.q.finish()

        gradO_cl = MyClTensor.from_np(be, self.gradO)

        gradI = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        gradI_cl = MyClTensor.from_np(be, gradI)

        gradW = np.zeros((input_filters,3,3,output_filters), dtype=np.float32)
        gradW_cl = MyClTensor.from_np(be, gradW)
        
        conv.deltas = gradI_cl
        conv.dW = gradW_cl

        self.O = O
        self.q = be.q
        self.conv = conv
        self.gradW = gradW
        self.gradI = gradI

        self.I_cl = I_cl
        self.O_cl = O_cl
        self.gradO_cl = gradO_cl
        self.gradW_cl = gradW_cl
        self.gradI_cl = gradI_cl

    def sync(self):
        self.q.finish()

    def fprop(self):
        self.conv.fprop(self.I_cl)

    def bprop(self):
        self.conv.bprop(self.gradO_cl)

    def getO(self):
        self.O_cl.to_host()
        return self.O

    def getGradW(self):
        self.gradW_cl.to_host()
        return self.gradW

    def getGradI(self):
        self.gradI_cl.to_host()
        return self.gradI


# tests using opencl backend, in winogradcl branch

import sys
import time
import importlib
import numpy as np
import pyopencl as cl
from neoncl import api


mf = cl.mem_flags

class Test(object):
    def __init__(self, batch_size, its, layer_def, W, I, gradO):
        input_filters = layer_def['Ci']
        output_filters = layer_def['Co']
        image_size = layer_def['iW']
        assert layer_def['iH'] == image_size
        assert layer_def['kH'] == layer_def['kW'] == 3

        assert input_filters >= 4

        self.W = W
        self.I = I
        self.gradO = gradO

        gpu_idx = 0

        platforms = cl.get_platforms()
        i = 0
        for platform in platforms:
           gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
           if gpu_idx < i + len(gpu_devices):
               ctx = cl.Context(devices=[gpu_devices[gpu_idx - i]])
               break
           i += len(gpu_devices)

        print('cl_context', ctx)
        #ctx = cl.create_some_context()
        q = cl.CommandQueue(ctx)

        W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
        I_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=I)

        # W_cl = MyClTensor.from_np(be, self.W)
#        print('self.I', self.I)
        # I_cl = MyClTensor.from_np(be, self.I)

        # conv = Convolution((3, 3, output_filters), strides=1, padding=1, be=be) #, init=init)
        # conv.configure((input_filters,image_size, image_size))
        # conv.W = W_cl

        O = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O)
        # O_cl = MyClTensor.from_np(be, O)
        # conv.outputs = O_cl

        # conv.fprop(I_cl)
        # be.q.finish()

        # gradO_cl = MyClTensor.from_np(be, self.gradO)
        gradO_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradO)

        gradI = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        gradI_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradI)
        # gradI_cl = MyClTensor.from_np(be, gradI)

        gradW = np.zeros((input_filters,3,3,output_filters), dtype=np.float32)
        gradW_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradW)
        # gradW_cl = MyClTensor.from_np(be, gradW)
        
        # conv.deltas = gradI_cl
        # conv.dW = gradW_cl

        self.convolver = api.Convolver(ctx, batch_size, input_filters, output_filters,
            layer_def['kH'], layer_def['kW'], layer_def['iH'], layer_def['iW'],
            layer_def['kH'] // 2, layer_def['kW'] // 2)
        self.scratch_size = self.convolver.getScratchSize()
        self.scratch = np.zeros(self.scratch_size, dtype=np.float32)
        self.scratch_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.scratch)

        self.O = O
        self.q = q
        self.W_cl = W_cl
        # self.convolver = convo
        self.gradW = gradW
        self.gradI = gradI

        self.I_cl = I_cl
        self.O_cl = O_cl
        self.gradO_cl = gradO_cl
        self.gradW_cl = gradW_cl
        self.gradI_cl = gradI_cl

        self.ctx = ctx

    def sync(self):
        self.q.finish()

    def fprop(self):
        self.convolver.fprop(self.ctx, self.q, self.I_cl, self.W_cl, self.O_cl)

    def bprop(self):
        self.convolver.bprop_gradW(self.ctx, self.q, self.I_cl, self.gradO_cl, self.gradW_cl)
        self.convolver.bprop_gradI(self.ctx, self.q, self.gradO_cl, self.W_cl, self.gradI_cl, self.scratch_cl)
        pass
        # self.conv.bprop(self.gradO_cl)

    def getO(self):
        # self.O_cl.to_host()
        cl.enqueue_copy(self.q, self.O, self.O_cl)
        return self.O

    def getGradW(self):
        # self.gradW_cl.to_host()
        cl.enqueue_copy(self.q, self.gradW, self.gradW_cl)
        return self.gradW

    def getGradI(self):
        # self.gradI_cl.to_host()
        cl.enqueue_copy(self.q, self.gradI, self.gradI_cl)
        return self.gradI


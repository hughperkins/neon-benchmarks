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
        assert layer_def['iH'] == layer_def['iW']
        assert layer_def['kH'] == layer_def['kW']# == 3
        assert layer_def['padH'] == layer_def['padW'] # == 1

        # assert layer_def['kH'] == 3
        # assert layer_def['padH'] == 1
        # assert layer_def['Ci'] >= 4

        input_filters = layer_def['Ci']
        output_filters = layer_def['Co']
        image_size = layer_def['iW']
        pad = layer_def['padW']
        stride = layer_def['dW']
        kH = layer_def['kH']
        kW = layer_def['kW']

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
        q = cl.CommandQueue(ctx)

        W_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=W)
        I_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=I)
        gradO_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradO)

        O = np.zeros((image_size * image_size * output_filters, batch_size), dtype=np.float32)
        O_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=O)

        gradI = np.zeros((input_filters,image_size, image_size,batch_size), dtype=np.float32)
        gradI_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradI)

        gradW = np.zeros((input_filters,kH,kW,output_filters), dtype=np.float32)
        gradW_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=gradW)

        self.convolver = api.Convolver(ctx, batch_size, input_filters, output_filters,
            layer_def['kH'], layer_def['kW'], layer_def['iH'], layer_def['iW'],
            padH=layer_def['padH'], padW=layer_def['padW'],
            dH=layer_def['dH'], dW=layer_def['dW'])
        self.scratch_size = self.convolver.getScratchSize()
        self.scratch = np.zeros(self.scratch_size, dtype=np.float32)
        self.scratch_cl = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.scratch)

        self.O = O
        self.q = q
        self.W_cl = W_cl
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
        self.convolver.fprop(self.q, self.I_cl, self.W_cl, self.O_cl)

    def bprop(self):
        self.convolver.bprop_gradW(self.q, self.I_cl, self.gradO_cl, self.gradW_cl)
        self.convolver.bprop_gradI(self.q, self.gradO_cl, self.W_cl, self.gradI_cl, self.scratch_cl)

    def getO(self):
        cl.enqueue_copy(self.q, self.O, self.O_cl)
        return self.O

    def getGradW(self):
        cl.enqueue_copy(self.q, self.gradW, self.gradW_cl)
        return self.gradW

    def getGradI(self):
        cl.enqueue_copy(self.q, self.gradI, self.gradI_cl)
        return self.gradI


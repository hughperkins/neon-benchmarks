import cpuref
import numpy as np

Ci = 32
Co = 32
N = 32
kH = 3
kW = 3
iW = 8
iH = 8
dW = 4
dH = 4
padW = 1
padH = 1
oH = (iH - kH + 2 * padH) // dH + 1
oW = (iW - kW + 2 * padW) // dW + 1

I = np.zeros((Ci, iH, iW, N), dtype=np.float32)
W = np.zeros((Co, kH, kW, Ci), dtype=np.float32)
gradO = np.zeros((Co, oH, oW, N), dtype=np.float32)
gradI = np.zeros((Ci, iH, iW, N), dtype=np.float32)

np.random.seed(123)
I[:] = np.random.randn(*I.shape)
W[:] = np.random.randn(*W.shape)
gradO[:] = np.random.randn(*gradO.shape)
#gradI[:] = np.random.randn(*gradI.shape)

#W[0, 1, 1, 0] = 1.234
#W[0, 0, 1, 0] = 1.111
#gradO[0, 1, 0, 0] = -0.5
#gradO[0, 0, 1, 0] = 2.0
#gradO[0, 1, 1, 0] = 3.0

stride = dW
pad = padW

for ih in range(iH):
    for iw in range(iW):
        gradI[0, ih, iw, 0] = cpuref.calcGradI(W=W, gradO=gradO, ci=0, ih=ih, iw=iw, n=0, pad=pad, stride=stride)

# print('gradI[0, :, :, 0]', gradI[0, :, :, 0])
print('gradI')
print(str(gradI[0, :, :, 0]).replace('\\n', '\n').replace('        ', ' '))


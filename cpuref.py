"""
Different from the backends, this will only compute for specific positions in O, gradW, or gradI,
for speed

We are reasonably sure it's correct, since it uses such a naive, but readable, implementation

It wont be very fast though, hence we just sample a few possible values, and check those

Layouts:
  I,gradI  ci ih iw n
  W,gradW  ci kh kw co
  O,gradO  co oh ow n

Where:
  ci: input channel
  co: output channel
  ih: input row
  iw: input col
  oh: output row
  ow: output col
  kh: filter row
  kw: filter col
  n: index into batch
"""

def check_gradW(I, W, gradO, gradW, ci, h, w, co, eps=1e-2):
#    eps = 1e4 #hack
    N = I.shape[3]
    iH = I.shape[1]
    iW = I.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

#    ih = h
#    iw = w
    kh = h
    kw = w
#    ci = c

    padw = 1
    padh = 1

    sum = 0

    for ow in range(oW):
        for oh in range(oH):
            ih = oh + kh - padh
            iw = ow + kw - padw
            for n in range(N):
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * gradO[co * iH * iW + oh * iW + ow, n]
                    sum += v
    cpu_value = sum
    gpu_value = gradW[ci, kh, kw, co]
#    print('gpu', gpu_value, 'cpu', cpu_value)
    diff = abs(cpu_value - gpu_value)
    print('check gradW ci=%s h=%s w=%s co=%s cpu %.4f gpu %.4f diff=%.4f' % (ci, h, w, co, cpu_value, gpu_value, diff))
    assert abs(cpu_value - gpu_value) < eps
    return cpu_value

def check_gradI(W, gradO, gradI, c, h, w, n, eps=1e-4):
    N = gradI.shape[3]
    iH = gradI.shape[1]
    iW = gradI.shape[2]
    Ci = W.shape[0]
    kH = W.shape[1]
    kW = W.shape[2]
    Co = W.shape[3]
    oH = iH # assuming padded, which it is
    oW = iW # assuming padded, which it is
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    ih = h
    iw = w
    ci = c

    padw = 1
    padh = 1

    sum = 0
    for co in range(Co):
        for kh in range(kH):
            for kw in range(kW):
                ow = iw - kw + padw
                oh = ih - kh + padh
                if ow >= 0 and oh >= 0 and ow < oW and oh < oH:
                    v = gradO[co * iH * iW + oh * iW + ow, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = gradI[c, ih, iw, n]
#    print('gpu', gpu_value, 'cpu', cpu_value)
    diff = abs(cpu_value - gpu_value)
    print('check gradI c=%s h=%s w=%s n=%s cpu %.4f gpu %.4f diff=%.4f' % (c, h, w, n, cpu_value, gpu_value, diff))
    assert abs(cpu_value - gpu_value) < eps
    return cpu_value

def check_O(gpu_O, W, I, c, h, w, n, eps=1e-4):
    Ci = W.shape[0]
    iH = I.shape[1]
    iW = I.shape[2]
    Co = W.shape[3]
    kH = W.shape[1]
    kW = W.shape[2]
#    print('Ci', Ci, 'iH', iH, 'iW', iW, 'Co', Co, 'kH', kH, 'kW', kW)

    co = c
    padw = 1
    padh = 1

    # we are going to apply entire kernel, over all input channels, to the input
    # image, in one location
    sum = 0
    for kw in range(kW):
        for kh in range(kH):
            for ci in range(Ci):
                ih = h + kh - padh
                iw = w + kw - padw
                if ih >= 0 and iw >= 0 and ih < iH and iw < iW:
                    v = I[ci, ih, iw, n] * W[ci, kh, kw, co]
                    sum += v
    cpu_value = sum
    gpu_value = gpu_O[c*iH*iW + h*iW + w,n]
    diff = abs(cpu_value - gpu_value)
    print('check O c=%s h=%s w=%s n=%s cpu %.4f gpu %.4f diff=%.4f' % (c, h, w, n, cpu_value, gpu_value, diff))
    assert abs(cpu_value - gpu_value) < eps
    return cpu_value


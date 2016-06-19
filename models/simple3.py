# just something that I can run on my laptop quickly...
# this has pad not kernelsize//2

def get_batchsize():
    return 32

def get_net():
    net = []
    layer = {'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3, 'padH': 0, 'padW': 0,
             'dH': 1, 'dW': 1}
    net.append(layer)
    return net


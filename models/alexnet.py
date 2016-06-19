import math

# ref: https://github.com/akrizhevsky/cuda-convnet2/blob/master/layers/layers-imagenet-1gpu.cfg
def get_net():
    net = []
    net.append({'Ci': 3, 'Co': 64, 'iH': 224, 'iW': 224, 'kH': 11, 'kW': 11, 'dH': 4, 'dW': 4,
                'padH': 0, 'padW': 0})
    net.append({'Ci': 64, 'Co': 192, 'iH': 27, 'iW': 27, 'kH': 5, 'kW': 5, 'dH': 1, 'dW': 1,
                'padH': 2, 'padW': 2})
    net.append({'Ci': 192, 'Co': 384, 'iH': 13, 'iW': 13, 'kH': 3, 'kW': 3, 'dH': 1, 'dW': 1,
                'padH': 1, 'padW': 1})
    net.append({'Ci': 384, 'Co': 256, 'iH': 13, 'iW': 13, 'kH': 3, 'kW': 3, 'dH': 1, 'dW': 1,
                'padH': 1, 'padW': 1})
    net.append({'Ci': 256, 'Co': 256, 'iH': 13, 'iW': 13, 'kH': 3, 'kW': 3, 'dH': 1, 'dW': 1,
                'padH': 1, 'padW': 1})
    return net

def get_batchsize():
    return 128


import math

def get_net():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    imageSize = 224
    channels = 3

    net = []
    conv_index = 0
    for i, op in enumerate(cfg):
        if isinstance(op, int):
            layer = {'Ci': channels, 'Co': op, 'iH': imageSize, 'iW': imageSize, 'kH': 3, 'kW': 3,
                     'dH': 1, 'dW': 1, 'padH': 1, 'padW': 1}
            net.append(layer)
            channels = op
            conv_index += 1
        elif op == 'M':
            imageSize = int(math.ceil(imageSize / 2))
        else:
            raise Exception('not implemented %s' % op)
    return net

def get_batchsize():
    return 64


import math

def get_net():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    imageSize = 224
    channels = 3

    net = []
    for i, op in enumerate(cfg):
        if isinstance(op, int):
            layer = {'Ci': channels, 'Co': op, 'iH': imageSize, 'iW': imageSize, 'kH': 3, 'kW': 3}
            layer['epsO'] = 1e-4
            layer['epsGradW'] = 1e-4
            layer['epsGradI'] = 1e-4
            net.append(layer)
            channels = op
        elif op == 'M':
            imageSize = int(math.ceil(imageSize / 2))
        else:
            raise Exception('not implemented %s' % op)
    return net

def get_batchsize():
    return 64

#def getEpsO():
#    return 1e-4

#def getEpsGradW():
#    return 1e-2

#def getEpsGradI():
#    return 1e-3


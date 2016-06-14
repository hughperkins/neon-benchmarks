import math

def get_net():
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    imageSize = 224
    channels = 3

    net = []
    conv_index = 0
    for i, op in enumerate(cfg):
        if isinstance(op, int):
            layer = {'Ci': channels, 'Co': op, 'iH': imageSize, 'iW': imageSize, 'kH': 3, 'kW': 3}
            layer['epsO'] = 1e-4
            layer['epsGradW'] = 1e-4
            layer['epsGradI'] = 1e-4
            if conv_index == 1:
                layer['epsGradW'] = 1e-2
            elif conv_index == 2:
                layer['epsGradW'] = 1e-3
            elif conv_index == 3:
                layer['epsGradW'] = 1e-3
                layer['epsGradI'] = 1e-3
            elif conv_index == 4:
                layer['epsGradW'] = 1e-3
            elif conv_index == 5:
                layer['epsO'] = 1e-3
                layer['epsGradW'] = 1e-3
                layer['epsGradI'] = 1e-3
            elif conv_index == 6:
                layer['epsO'] = 1e-3
                layer['epsGradW'] = 1e-3
                layer['epsGradI'] = 1e-3
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


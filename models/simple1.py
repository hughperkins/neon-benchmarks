# just something that I can run on my laptop quickly...

def get_batchsize():
    return 32

def get_net():
    net = []
    layer = {'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3}
    layer['epsO'] = 1e-4
    layer['epsGradW'] = 1e-3
    layer['epsGradI'] = 1e-4
    net.append(layer)
    return net


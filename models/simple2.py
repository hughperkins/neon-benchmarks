# just something that I can run on my laptop quickly...
# this has stride not 1

def get_batchsize():
    return 32

def get_net():
    net = []
    layer = {'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3, 'padH': 1, 'padW': 1,
             'dH': 4, 'dW': 4}
    net.append(layer)
    return net


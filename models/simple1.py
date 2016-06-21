# just something that I can run on my laptop quickly...

def get_batchsize():
    return 32

def get_net():
    net = []
    net.append({'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3, 'padH': 1, 'padW': 1,
             'dH': 1, 'dW': 1})
    net.append({'Ci': 32, 'Co': 32, 'iH': 8, 'iW': 8, 'kH': 3, 'kW': 3, 'padH': 1, 'padW': 1,
             'dH': 4, 'dW': 4})  # has stride > 1
    net.append({'Ci': 4, 'Co': 4, 'iH': 4, 'iW': 4, 'kH': 3, 'kW': 3, 'padH': 1, 'padW': 1,
             'dH': 4, 'dW': 4})  # smaller version of previous layer
    net.append({'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3, 'padH': 0, 'padW': 0,
             'dH': 1, 'dW': 1})  # zero padding
    return net


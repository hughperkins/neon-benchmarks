# just something that I can run on my laptop quickly...

def get_batchsize():
    return 32

def get_net():
    net = [
        {'Ci': 32, 'Co': 32, 'iH': 64, 'iW': 64, 'kH': 3, 'kW': 3}
    ]
    return net

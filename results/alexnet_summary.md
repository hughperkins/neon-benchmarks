# VGG A Summary

These results are for [Alexnet](../models/alexnet.py), on a Titan X.

## Nervana Neon CUDA/SASS Winograd kernels for Maxwell

[neon_maxwell](backends/neon_maxwell.py)
```
Layer 0: fprop=0.003 bprop=0.012 eps_O=4e-06 eps_gradW=5e-04 eps_gradI=2e-06
Layer 1: fprop=0.009 bprop=0.020 eps_O=9e-06 eps_gradW=4e-04 eps_gradI=3e-05
Layer 2: fprop=0.003 bprop=0.006 eps_O=5e-05 eps_gradW=3e-04 eps_gradI=8e-05
Layer 3: fprop=0.004 bprop=0.007 eps_O=5e-05 eps_gradW=7e-04 eps_gradI=1e-04
Layer 4: fprop=0.003 bprop=0.005 eps_O=1e-04 eps_gradW=3e-04 eps_gradI=6e-05
```

## Nervana Neon Kepler direct kernels, in CUDA

[neon_kepler](backends/neon_kepler.py)
```
Layer 0: SKIPPED
Layer 1: fprop=0.015 bprop=0.046 eps_O=9e-06 eps_gradW=2e-04 eps_gradI=3e-05
Layer 2: fprop=0.008 bprop=0.023 eps_O=2e-05 eps_gradW=3e-05 eps_gradI=5e-05
Layer 3: fprop=0.010 bprop=0.030 eps_O=4e-05 eps_gradW=3e-05 eps_gradI=2e-05
Layer 4: fprop=0.007 bprop=0.020 eps_O=3e-05 eps_gradW=2e-05 eps_gradI=2e-05
```

## OpenCL port of Nervana Neon Kepler direct kernels

[neoncl_direct](backends/neoncl_direct.py)
```
Layer 0: SKIPPED
Layer 1: fprop=0.016 bprop=0.049 eps_O=1e-05 eps_gradW=1e-04 eps_gradI=6e-05
Layer 2: fprop=0.008 bprop=0.024 eps_O=2e-05 eps_gradW=3e-05 eps_gradI=3e-05
Layer 3: fprop=0.010 bprop=0.031 eps_O=4e-05 eps_gradW=3e-05 eps_gradI=3e-05
Layer 4: fprop=0.007 bprop=0.021 eps_O=2e-05 eps_gradW=2e-05 eps_gradI=2e-05
```


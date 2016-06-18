# VGG A Summary

These results are for [VGG A](../models/vgga.py), on a Titan X.

## Nervana Neon CUDA/SASS Winograd kernels for Maxwell

```
Layer 0: fprop=0.004 bprop=0.036 eps_O=3e-07 eps_gradW=2e-03 eps_gradI=4e-05
Layer 1: fprop=0.012 bprop=0.032 eps_O=1e-05 eps_gradW=1e-02 eps_gradI=3e-05
Layer 2: fprop=0.009 bprop=0.021 eps_O=5e-05 eps_gradW=2e-03 eps_gradI=6e-05
Layer 3: fprop=0.017 bprop=0.036 eps_O=1e-04 eps_gradW=3e-03 eps_gradI=8e-05
Layer 4: fprop=0.007 bprop=0.016 eps_O=8e-05 eps_gradW=6e-04 eps_gradI=2e-04
Layer 5: fprop=0.015 bprop=0.031 eps_O=9e-05 eps_gradW=8e-04 eps_gradI=1e-04
Layer 6: fprop=0.005 bprop=0.010 eps_O=8e-05 eps_gradW=4e-04 eps_gradI=9e-05
Layer 7: fprop=0.005 bprop=0.010 eps_O=8e-05 eps_gradW=4e-04 eps_gradI=9e-05
```

## Nervana Neon Kepler direct kernels, in CUDA

```
Layer 0: SKIPPED
Layer 1: fprop=0.032 bprop=0.158 eps_O=9e-06 eps_gradW=1e-03 eps_gradI=2e-05
Layer 2: fprop=0.033 bprop=0.110 eps_O=2e-05 eps_gradW=3e-04 eps_gradI=2e-05
Layer 3: fprop=0.067 bprop=0.222 eps_O=2e-05 eps_gradW=3e-04 eps_gradI=3e-05
Layer 4: fprop=0.033 bprop=0.111 eps_O=3e-05 eps_gradW=1e-04 eps_gradI=9e-05
Layer 5: fprop=0.066 bprop=0.222 eps_O=4e-05 eps_gradW=9e-05 eps_gradI=5e-05
Layer 6: fprop=0.016 bprop=0.053 eps_O=4e-05 eps_gradW=2e-05 eps_gradI=3e-05
Layer 7: fprop=0.016 bprop=0.053 eps_O=4e-05 eps_gradW=2e-05 eps_gradI=3e-05
```

## OpenCL port of Nervana Neon Kepler direct kernels

```
Layer 0: SKIPPED
Layer 1: fprop=0.039 bprop=0.173 eps_O=1e-05 eps_gradW=8e-04 eps_gradI=1e-05
Layer 2: fprop=0.039 bprop=0.124 eps_O=1e-05 eps_gradW=4e-04 eps_gradI=2e-05
Layer 3: fprop=0.073 bprop=0.237 eps_O=3e-05 eps_gradW=2e-04 eps_gradI=4e-05
Layer 4: fprop=0.038 bprop=0.125 eps_O=2e-05 eps_gradW=5e-05 eps_gradI=2e-05
Layer 5: fprop=0.073 bprop=0.238 eps_O=3e-05 eps_gradW=1e-04 eps_gradI=4e-05
Layer 6: fprop=0.022 bprop=0.069 eps_O=5e-05 eps_gradW=2e-05 eps_gradI=7e-05
Layer 7: fprop=0.021 bprop=0.067 eps_O=5e-05 eps_gradW=2e-05 eps_gradI=7e-05
```


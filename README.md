# neon-benchmarks
benchmarks for neon, for various backends, including OpenCL direct.

This is a separate repo, so it can be used against both repos

Pluggable test backend lets tests run against both

## How to use: neoncl

* install neoncl https://github.com/hughperkins/neonCl-underconstruction
* install neon-benchmarks, eg:
```
pip install -e ./
```
* run:
```
neon_benchmarks.py
```

## How to use: neon base

* Install neon base from https://github.com/nervanasystems/neon
* Install neon-benchmarks.  From the `neon` directory:
```
source .venv/bin/activate
```
* Then, from the `neon-benchmarks` directory do:
```
pip install -e ./
```
* run, Maxwell Winograd SASS kernels:
```
neon_benchmarks.py --backend sass_winograd
```
* run, Kepler direct CUDA kernels:
```
neon_benchmarks.py --backend cuda_direct
```
## Model selection

There are different network models in the [models](models) directory.  Simply use the `--model` parameter to choose one, eg:
```
neon_benchmarks.py --model simple1  # this is mostly for debugging / dev
neon_benchmarks.py --model alexnet
neon_benchmarks.py --model vgga
```

## Results

* [VGG A on Titan X](results/vgga_summary.md)


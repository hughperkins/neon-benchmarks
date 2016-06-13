# neon-benchmarks
benchmarks for neon, both cuda and OpenCL version

This is a separate repo, so it can be used against both repos

Pluggable test backend lets tests run against both

## How to use: winogradcl

* first, install winogradcl https://github.com/hughperkins/winogradCl-underconstruction
* install this repo, eg:
```
cd neon-benchmarks
pip install -e ./
```
* run:
```
neon_benchmarks.py
```
That's it! :-)

## How to use: neon base

* Install neon base from https://github.com/nervanasystems/neon
* install this repo, eg:
```
cd neon-benchmarks
pip install -e ./
```
* run:
```
neon_benchmarks.py --backend neonbase
```

## Model selection

There are different network models in the [models](models) directory.  Simply use the `--model` parameter to choose one, eg:
```
neon_benchmarks.py --model simple1
```


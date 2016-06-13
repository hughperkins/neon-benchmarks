# neon-benchmarks
benchmarks for neon, both cuda and OpenCL version

This is a separate repo, so it can be used against both repos

Pluggable test backend lets tests run against both

## How to use: winogradcl

* install winogradcl https://github.com/hughperkins/winogradCl-underconstruction
* install neon-benchmarks, eg:
```
pip install -e ./
```
* run:
```
neon_benchmarks.py
```
That's it! :-)

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
* run:
```
neon_benchmarks.py --backend neonbase
```
(note that neon base needs python2.7 currently, so you'll need to create a python2.7 virtualenvironment,

## Model selection

There are different network models in the [models](models) directory.  Simply use the `--model` parameter to choose one, eg:
```
neon_benchmarks.py --model simple1
```


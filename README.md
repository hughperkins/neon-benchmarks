# neon-benchmarks
benchmarks for neon, both cuda and OpenCL version

This is a separate repo, so it can be used against both repos

Pluggable test backend lets tests run against both

## How to use: winogradcl

* first, install winogradcl https://github.com/hughperkins/winogradCl-underconstruction
* now, using the same virtualenv you just set up, simply run, from this `neon-benchmarks` repo:
```
cd neon-benchmarks
python run_benchmarks.py
```
That's it! :-)

## How to use: neon base

* Install neon base from https://github.com/nervanasystems/neon
* Using the same venv you just created, ie do `source .venv/bin/activate` from the `neon` directory, do:
```
cd neon-benchmarks
python run_benchmarks.py --backend neonbase
```


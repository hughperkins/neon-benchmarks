#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc. Hugh Perkins 2016
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import os
from setuptools import setup, find_packages, Command
import subprocess

# Define version information
VERSION = '0.0.1'
FULLVERSION = VERSION
write_version = True

try:
    pipe = subprocess.Popen(["git", "rev-parse", "--short", "HEAD"],
                            stdout=subprocess.PIPE)
    (so, serr) = pipe.communicate()
    if pipe.returncode == 0:
        FULLVERSION += "+%s" % so.strip().decode("utf-8")
except:
    pass

if write_version:
    txt = "\"\"\"\n%s\n\"\"\"\nVERSION = '%s'\nSHORT_VERSION = '%s'\n"
    fname = os.path.join(os.path.dirname(__file__), '.', 'version.py')
    a = open(fname, 'w')
    try:
        a.write(txt % ("Project version information.", FULLVERSION, VERSION))
    finally:
        a.close()


setup(name='neon-benchmarks',
      version=VERSION,
      description="Benchmarks for nervana neon kernels, including OpenCL port",
      long_description=open('README.md').read(),
      author='Hugh Perkins',
      author_email='hughperkins@gmail.com',
      url='http://github.com/hughperkins/neon-benchmarks',
      license='License :: OSI Approved :: Apache Software License',
      scripts=['neon_benchmarks.py'],
      packages=find_packages(exclude=["tests"]),
      package_data={'winogradcl': []},
      classifiers=['Development Status :: 2 - Beta',
                   'Intended Audience :: Developers',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: Apache Software License',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: ' +
                   'Artificial Intelligence',
                   'Topic :: Scientific/Engineering :: Information Analysis',
                   'Topic :: System :: Distributed Computing'])

#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import,print_function
from setuptools import setup,find_packages
import os
setup(name="FBG_3DGen",
     version='0.1',
     description='3D Fragment based molecular graph generative model',
     author='Mingyuan Xu',
     license='GPL3',
     packages=find_packages(),
     zip_safe=False,
     include_package_data=True,
     )


#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Stephan Sahm <Stephan.Sahm@gmx.de>'

import os
import shutil
from setuptools import setup, find_packages
from distutils.command.clean import clean as Clean


class CleanCmd(Clean):

    description = "Cleans ..."

    def run(self):

        Clean.run(self)

        if os.path.exists('build'):
            shutil.rmtree('build')

        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (filename.endswith('.so') or
                    filename.endswith('.pyd') or
                    filename.endswith('.pyc') or
                    filename.endswith('_wrap.c') or
                    filename.startswith('wrapper_') or
                    filename.endswith('~')):
                        os.unlink(os.path.join(dirpath, filename))

            for dirname in dirnames:
                if dirname == '__pycache__' or dirname == 'build':
                    shutil.rmtree(os.path.join(dirpath, dirname))
                if dirname == "pymscons.egg-info":
                    shutil.rmtree(os.path.join(dirpath, dirname))

setup(
    name='theano_models',
    version='0.1.0',
    description='abstract helpers to simplify working with theano expressions in an machine learning context',
    author=__author__,
    author_email='Stephan.Sahm@gmx.de',
    license='to be announced',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        "schlichtanders>=0.1.0",
        "numpy>=1.10.2",
        "scipy>=0.13.3",
        "theano>=0.9.0.dev0",
        "climin>=pre-0.1",
        "six>=1.10.0",
    ],
    dependency_links=[
        "git+https://github.com/schlichtanders/schlichtanders.git#egg=schlichtanders-0.1.0",
        "git+https://github.com/BRML/climin.git#egg=climin-pre-0.1"
    ],
    # include_package_data=True,  # should work, but doesn't, I think pip does not recognize git automatically
    package_data={
        'data': ['*/*'],
    },
    cmdclass={'clean': CleanCmd}
)

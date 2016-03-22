''' Thermopy setup script'''

from setuptools.command.test import test as TestCommand
import sys

from setuptools import setup

def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()

setup(
    author = "Lukasz Mentel",
    author_email = "lmmentel@gmail.com",
    description = "Functions for calculating thermochemistry with anharmonic corrections",
    include_package_data = True,
    install_requires = [
        'numpy >= 1.7',
    ],
    entry_points = {
        'console_scripts' : [
            'thermopy = thermo.thermo:main',
        ]
    },
    license = open('LICENSE.rst').read(),
    long_description = readme(),
    name = 'thermopy',
    packages = ['thermo'],
    url = 'https://bitbucket.org/lukaszmentel/thermopy/',
    version = '0.1.0',
    classifiers = [
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

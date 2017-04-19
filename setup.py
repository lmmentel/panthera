
''' panther setup script'''

from setuptools import setup


def readme():
    '''Return the contents of the README.md file.'''
    with open('README.md') as freadme:
        return freadme.read()

setup(
    author = "Lukasz Mentel",
    author_email = "lmmentel@gmail.com",
    description = "Package for calculating thermochemistry with anharmonic corrections",
    include_package_data = True,
    entry_points = {
        'console_scripts' : [
            'panther = panther.panther:main',
            'plotmode = panther.cli:plotmode_cli',
            'writemodes = panther.cli:write_modes_cli',
        ]
    },
    license = open('LICENSE.rst').read(),
    long_description = readme(),
    name = 'panther',
    packages = ['panther'],
    url = 'https://bitbucket.org/lukaszmentel/panther/',
    version = '0.5.1',
    classifiers = [
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)

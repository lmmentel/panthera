""" panthera setup script"""

from setuptools import setup


def readme():
    """Return the contents of the README.md file."""
    with open("README.md") as freadme:
        return freadme.read()


def from_requirements():
    "Return a list of requirements from a file"
    with open("panthera_requirements.txt", "r") as freq:
        return freq.read().splitlines()


setup(
    author="Lukasz Mentel",
    author_email="lmmentel@gmail.com",
    description="Package for calculating thermochemistry with anharmonic corrections",
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "panthera = panthera.panthera:main",
            "plotmode = panthera.cli:plotmode_cli",
            "writemodes = panthera.cli:write_modes_cli",
        ]
    },
    install_requires=from_requirements(),
    license=open("LICENSE.rst").read(),
    long_description=readme(),
    name="panthera",
    packages=["panthera"],
    url="https://bitbucket.org/lukaszmentel/panthera/",
    version="0.5.2",
    classifiers=[
        "Environment :: Console",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)

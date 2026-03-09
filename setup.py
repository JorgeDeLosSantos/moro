from setuptools import setup
import os

dir_setup = os.path.dirname(os.path.realpath(__file__))

version = {}
with open(os.path.join(dir_setup, "moro", "version.py")) as f:
    exec(f.read(), version)
    
__version__ = version["__version__"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='moro',
      version=__version__,
      description='Python library for kinematic and dynamic modeling of robots',
      author='Pedro Jorge De Los Santos',
      author_email='delossantosmfq@gmail.com',
      keywords=["Robotics","Kinematics","Dynamics"],
      licence='MIT',
      url='https://github.com/JorgeDeLosSantos/moro',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['moro',],
      install_requires=['sympy','matplotlib'],
      classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: Implementation :: CPython",
      ]
      )

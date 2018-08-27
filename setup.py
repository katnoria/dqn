#!/usr/bin/env python

from setuptools import setup, Command, find_packages


with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='dqn_navigation_mlagents',
      version='0.1.0',
      description='DQN navigation for Unity Machine Learning Agents',
      license='Apache License 2.0',
      author='Katnoria',      
      url='https://github.com/Unity-Technologies/ml-agents',
      packages=find_packages(),
      install_requires = required,
      long_description= ("Implementation of agents that use Unity Machine Learning environment")
     )

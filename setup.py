#!/usr/bin/env python

from setuptools import setup, find_packages
import os

# Prepare long description using existing docs
long_description = ""
this_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name="AMACollision",
    version='0.0.1',
    description='a multi-agent based ADS testing framework for generating high realism critical driving scenarios'
    'with Carla simulator',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alanshuo123/AMACollision.git',
    author='Shuo Tian',
    author_email='tianshuo@nuaa.edu.cn',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    python_requires='>=3.0',
    install_requires=[
        'gym', 'carla==0.9.13', 'GPUtil', 'pygame', 'opencv-python', 'networkx'
    ],
    extras_require={'test': ['tox', 'pytest', 'pytest-xdist', 'tox']},
    keywords='multi-agent learning environments connected autonomous driving '
    'OpenAI Gym CARLA',
    project_urls={
        'Source': 'https://github.com/alanshuo123/AMACollision.git',
        'Report bug': 'https://github.com/alanshuo123/AMACollision/issues',
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers', 'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ])

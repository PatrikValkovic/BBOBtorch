###############################
#
# Created by Patrik Valkovic
# 3/6/2021
#
###############################

from setuptools import setup

v = '2.0.0'
# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    README = f.read()

setup(
    name="BBOBtorch",
    version="1.0.0",
    description="Vectorized BBOB functions in torch",
    url="https://github.com/PatrikValkovic/BBOBtorch",
    download_url='https://github.com/PatrikValkovic/BBOBtorch/archive/v' + v + '.tar.gz',
    long_description=README,
    long_description_content_type="text/markdown",
    author="Patrik Valkovic",
    license="MIT",
    packages=["bbobtorch"],
    install_requires=[
        "torch",
    ],
)
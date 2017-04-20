"""
 Setup tools configuration for installing this package
"""
from setuptools import setup, find_packages

setup(name='kusanagi',
      version='0.1',
      description='A modular RL library',
      url='http://github.com/juancamilog/kusanagi',
      author='Juan Camilo Gamboa Higuera',
      author_email='juancamilog@gmail.com',
      license='MIT',
      packages=find_packages(exclude=['examples', 'thirdparty', 'doc', 'test']),
      install_requires=['theano', 'lasagne', 'pyserial', 'matplotlib'],
     )

from setuptools import find_packages
from setuptools import setup

setup(
    name= 'nn',
    version= '0.0.1',
    author= 'Grace Ramey',
    author_email= 'Grace.Ramey@ucsf.edu',
    packages= find_packages(),
    description= 'Neural network class and data preprocessor',
	install_requires= ['pytest', 'typing', 'numpy', 'scikit-learn', 'sklearn', 'numpy.typing']
)
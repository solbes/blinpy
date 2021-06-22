import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fh:
    long_description = fh.read()

setup(
    name='blinpy',
    version='0.1.5',
    packages=['blinpy'],
    url='https://github.com/solbes/blinpy',
    download_url = 'https://github.com/solbes/blinpy/archive/refs/tags/0.1.4.1.tar.gz',
    license='MIT',
    author='Antti Solonen',
    author_email='antti.solonen@gmail.com',
    description='Bayesian Linear Models in Python',
    keywords=['bayes', 'linear', 'gam'],
    install_requires=['numpy', 'pandas', 'jsonpickle', 'scipy', 'quadprog'],
    extras_require={
        'dev': ['pytest']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)

from setuptools import setup

setup(
    name='blinpy',
    version='0.1.2',
    packages=['blinpy'],
    url='https://github.com/solbes/blinpy',
    download_url = 'https://github.com/solbes/blinpy/archive/refs/tags/0.1.2.tar.gz',
    license='MIT',
    author='Antti Solonen',
    author_email='antti.solonen@gmail.com',
    description='Bayesian Linear Models in Python',
    keywords=['bayes', 'linear', 'gam'],
    install_requires=['numpy', 'pandas', 'jsonpickle'],
    extras_require={
        'dev': ['pytest']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Data Scientists',
        'Topic :: Data Science :: Linear Models',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
    ]
)

from setuptools import setup

setup(
    name='blinpy',
    version='0.1',
    packages=['blinpy'],
    url='https://github.com/solbes/blinpy',
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

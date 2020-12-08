import os
from setuptools import setup

os.environ['CVXOPT_BUILD_GLPK'] = '1'

setup(
    name='intvalpy',
    version='1.4.4',
    description='Interval library in Python',
    author='Андросов Артем Станиславович, Шарый Сергей Петрович',
    author_email='artem.androsov@gmail.com, shary@ict.nsc.ru',
    packages=['intvalpy'],
    install_requires=['Pillow',
                      'collection',
                      'cycler',
                      'joblib',
                      'kiwisolver',
                      'matplotlib',
                      'numpy',
                      'pip',
                      'pyparsing',
                      'python-dateutil',
                      'scipy',
                      'setuptools',
                      'six',
                      'wheel']
    )

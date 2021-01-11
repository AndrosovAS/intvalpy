import os
from setuptools import setup

os.environ['CVXOPT_BUILD_GLPK'] = '1'

INSTALL_REQUIRES = [
    'collection',
    'joblib',
    'matplotlib',
    'numpy',
    'scipy',
    'six',
    'wheel'
]

setup(
    name='intvalpy',
    version='1.4.6',
    description='Interval library in Python',
    author='Андросов Артем Станиславович, Шарый Сергей Петрович',
    author_email='artem.androsov@gmail.com, shary@ict.nsc.ru',
    url='https://github.com/Maestross/intvalpy',
    packages=['intvalpy'],
    install_requires=INSTALL_REQUIRES
)

import io
import os
from setuptools import setup, find_packages, Extension
import numpy as np


os.environ['CVXOPT_BUILD_GLPK'] = '1'

readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with io.open(readme_file, mode='r', encoding='utf-8') as f:
    README = f.read()

INSTALL_REQUIRES = [
    'matplotlib',
    'numpy',
    'cvxopt',
    'mpmath',
    'pandas',
    'cython'
]

setup(
    name='intvalpy',
    version='2.0.2',
    description='IntvalPy -- a Python interval computation library',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT License',
    keywords=[
        'interval',
        'machine learning',
        'data fitting',
        'inequality visualization',
        'optimal solutions',
        'math'
    ],
    author='Androsov, Artem Stanislavovich and Shary, Sergey Petrovich',
    author_email='astandrosov@yandex.ru, shary@ict.nsc.ru',
    url='https://github.com/AndrosovAS/intvalpy',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    ext_modules=[
        Extension(
            "intvalpy.kernel.interval_arithmetics",
            ["intvalpy/kernel/interval_arithmetics.pyx"],
            include_dirs=[np.get_include()]
        )
    ],
    include_dirs=[np.get_include()],
    zip_safe=False,
)

import io
import os
import re
from setuptools import setup, find_packages, Extension
import numpy as np


os.environ['CVXOPT_BUILD_GLPK'] = '1'


readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with io.open(readme_file, mode='r', encoding='utf-8') as f:
    README = f.read()


def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'intvalpy', '_version.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', content)
        if match:
            return match.group(1)
    raise RuntimeError("Unable to find version string.")
__version__ = get_version()


INSTALL_REQUIRES = [
    'matplotlib',
    'numpy',
    'cvxopt',
    'pandas',
    'cython'
]

setup(
    name='intvalpy',
    version=__version__,
    description='IntvalPy -- a Python interval computation library',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT License',
    license_files=['LICENSE.txt'],
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

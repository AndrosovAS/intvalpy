import io
import os
from setuptools import setup, find_packages

os.environ['CVXOPT_BUILD_GLPK'] = '1'

readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with io.open(readme_file, mode='r', encoding='utf-8') as f:
    README = f.read()

INSTALL_REQUIRES = [
    'matplotlib',
    'numpy',
    'scipy',
    'cvxopt',
    'mpmath'
]

setup(
    name='intvalpy',
    version='1.5.4',
    description='An interval library in Python that uses classical interval ' + \
                'arithmetic and Kauсher arithmetic + Kahan division in some functions',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT License',
    keywords=[
        'Interval',
        'inequality visualization',
        'optimal solutions',
        'math',
        'range'
    ],
    author='Андросов Артем Станиславович, Шарый Сергей Петрович',
    author_email='artem.androsov@gmail.com, shary@ict.nsc.ru',
    url='https://github.com/AndrosovAS/intvalpy',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES
)

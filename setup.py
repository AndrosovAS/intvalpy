import io
import os
from setuptools import setup, find_packages

os.environ['CVXOPT_BUILD_GLPK'] = '1'

readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
with io.open(readme_file, mode='r', encoding='utf-8') as f:
    README = f.read()

INSTALL_REQUIRES = [
    'joblib',
    'matplotlib',
    'numpy',
    'scipy'
]

setup(
    name='intvalpy',
    version='1.4.6',
<<<<<<< HEAD
    description='Interval library in Python using classical interval ' + \
                'arithmetic + Kahan division in some functions',
    long_description=README,
    long_description_content_type='text/markdown',
    license='MIT License',
    keywords=[
        'Interval arithmetic',
        'inequality visualization',
        'optimal solutions'
    ],
=======
    description='Interval library in Python',
>>>>>>> 9c702db78dead9612ff7b18dac2f1ae9b8ba2ac2
    author='Андросов Артем Станиславович, Шарый Сергей Петрович',
    author_email='artem.androsov@gmail.com, shary@ict.nsc.ru',
    url='https://github.com/AndrosovAS/intvalpy',
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES
)

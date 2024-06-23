from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='COVID-19-BioMarker',
    version='1.1.0',
    description='A study of Covid-19 Gene Biomarkers',
    long_description='A study of Covid-19 Gene Biomarkers',
    url='https://github.com/sayan1886/CAM',
    author='Sayan Chatterjee',
    author_email='sayan1886@gmail.com',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    keywords='keras scikit-learn imblearn shap lime xgboost xai explanations gene biomarker covid-19',
    license='MIT',
    install_requires=['matplotlib', 'pandas', 'numpy', 'tensorflow', 'scikit-learn', 'imbalanced-learn', 'lightgbm', 'xgboost', 'lime', 'shap' ],
)
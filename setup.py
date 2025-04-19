# Rudra/setup.py
from setuptools import setup, find_packages

setup(
    name='rudra', # Your package name
    version='0.1.0', # Initial version
    packages=find_packages(), # Automatically find packages like 'rudra', 'rudra.common', 'rudra.tree'
    install_requires=[
        # List your dependencies here from requirements.txt
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'scikit-learn>=1.0.0',
    ],
    # Optional: author, description, etc.
    author='RUDRA',
    author_email='kulkarnibruhat@gmail.com',
    description='A preprocessing library for various ML model types.',
)
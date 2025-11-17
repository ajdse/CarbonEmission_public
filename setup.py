from setuptools import setup, find_packages

setup(
    name='modeltoolkit-carbonemission',  # A unique, generic name for the package
    version='0.1.0',             # Standard initial version
    author='Abhishek Jaiswal',
    description='A reusable, modular library for data processing and modeling pipelines.',
    # Automatically find the 'modeltoolkit' directory and treat it as a package
    packages=find_packages(),
    # List the key dependencies required to run your code
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.23.0',
        'xgboost>=1.3.0',
        # Add any other libraries used in model_toolkit.py
    ],
    # Configuration for development/testing
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
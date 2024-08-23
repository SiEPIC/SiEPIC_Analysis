from setuptools import setup, find_packages


requirements = ['setuptools>=70.2.0', 'numpy>=1.26.4', 'matplotlib>=3.8.3', 'pandas>=2.1.3', 'scipy>=1.11.3',
                'PyYAML>=6.0.1', 'pyPDF2>=3.0.1', 'reportlab>=4.0.7', 'requests>=2.31.0']

setup(
    name='analysis_package',
    version='1.0',
    description='description to be filled',
    author='Tenna Yuan',
    author_email='tenna@student.ubc.ca',
    packages=find_packages(),
    install_requires=requirements
)

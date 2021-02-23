from setuptools import setup

setup(
    name='mylib',
    version='0.1',
    packages=['ml-framework'],
    package_dir={"ml-framework": "ml-framework"},
    url='',
    license='MIT',
    author='Amit Maindola',
    author_email='maindola.amit@gmail.com',
    description='My custom ML Framework having Library of helper methods for ML and DL work',
    install_requires=["numpy",
                      "pandas",
                      "matplotlib",
                      "seaborn",
                      "sklearn",
                      "emoji",
                      "nltk"]
)

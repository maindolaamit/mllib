from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='ml_framework',
    version='0.1',
    author='Amit Maindola',
    author_email='maindola.amit@gmail.com',
    description='My custom ML Framework having Library of helper methods for ML and DL work',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/maindolaamit/mllib',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=["numpy", "pandas", "matplotlib", "seaborn", "sklearn",
                      "emoji", "nltk", "lightgbm", "xgboost"]
)

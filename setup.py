from setuptools import setup, find_packages

setup(
    name="because_py",
    version="0.1.0",
    description="Python NumPyro backend for the R 'because' package",
    author="because-pkg",
    url="https://github.com/because-pkg/because_py",
    packages=find_packages(),
    install_requires=[
        "numpyro",
        "jax",
        "jaxlib",
        "numpy"
    ],
    python_requires=">=3.8",
)

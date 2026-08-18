from setuptools import setup, find_packages

setup(
    name="genecs",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "bposd==2.1",
        "ldpc==2.4.1",
        "matplotlib==3.10.8",
        "mip==1.17.6",
        "networkx==3.4.2",
        "numpy==2.2.6",
        "pandas==2.3.3",
        "scipy==1.15.3",
    ],
)
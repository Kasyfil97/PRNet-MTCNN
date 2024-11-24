# setup.py
from setuptools import setup, find_packages

setup(
    name="prnet",
    version="0.0.1",
    author="kasyfil",
    author_email="kasyfil.albar97@gmail.com",
    description="a package from prnet that can be used to get 3D face reconstruction from a single image",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Kasyfil97/prnet",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Daftar dependensi package Anda
        "numpy==2.0.2",
        "tensorflow==2.18.0",
        "mtcnn==1.0.0",
        "scipy==1.14.1",
        "scikit-image==0.24.0"
    ],
    include_package_data=True,
    package_data={
        "prnet": ["weights/*"]
    }
)
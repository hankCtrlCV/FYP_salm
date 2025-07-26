# setup.py
from setuptools import setup, find_packages

setup(
    name="FYP_salm",
    version="0.1",
    packages=find_packages(),
    install_requires=["gtsam"],  # 如果你在 PyPI 上能 pip install gtsam
)

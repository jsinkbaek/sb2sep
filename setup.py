from setuptools import find_packages, setup

setup(
    name="sb2sep",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

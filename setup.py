from setuptools import setup, find_packages

setup(
    name="rltoolbox_components",
    version="0.1.0",
    author="Luca Mignatti",
    author_email="lucamignatti@icloud.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    description="Components for RLtoolbox",
)

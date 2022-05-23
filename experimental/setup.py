from setuptools import setup

setup(
    name="wgit",
    packages=["wgit"],
    version="1.0.0",
    description="WeiGit for checkpoint tracking",
    author="Riyasat Ohib and Min Xu",
    author_email="riyasat.ohib@gatech.edu",
    entry_points={"console_scripts": ["wgit = wgit:main"]},
)

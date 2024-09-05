from setuptools import setup

with open("requirements.txt") as file:
    content = file.readlines()

requirements = [x.strip() for x in content]

setup(name="build-api",
      packages=["build-api"],
      install_requires=requirements)

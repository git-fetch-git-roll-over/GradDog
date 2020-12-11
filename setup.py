import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="GradDog",
    version="1.3.0",
    description="Perform automatic differentiation (final project CS107)",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/git-fetch-git-roll-over/GradDog.git",
    author="Peyton Benac, Max Cembalest, Seeam Shahid Noor, Ivan Shu",
    author_email="pbenac@college.harvard.edu",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages= ["graddog"],
    include_package_data=True,
    install_requires=["numpy", "pytest", "pandas", "matplotlib"],
)

"""Setup script for neuroimaging pipeline."""

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="neuroimaging-pipeline",
    version="1.0.0",
    author="Shristee Pandey",
    author_email="pande.shris2000@gmail.com",
    description="Professional neuroimaging data analysis pipeline for fMRI, EEG, and structural MRI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/shristeepandey/neuroimaging-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
)

from setuptools import setup, find_packages

setup(
    name="beak",
    version="0.1.0",
    description="A toolkit for biophysical and evolutionary association modeling",
    author="Micah Olivas",
    author_email="micah5051olivas@gmail.com",
    license="MIT",
    packages=find_packages(where="src", exclude=["tests*", "examples*", "docs*"]),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "biopython"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Operating System :: OS Independent"
    ],
    python_requires=">=3.7"
)

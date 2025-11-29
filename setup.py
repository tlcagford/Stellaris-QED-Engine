from setuptools import setup, find_packages

setup(
    name="stellaris-qed-engine",
    version="0.1.0",
    author="Tony Eugene Ford",
    author_email="tlcagford@gmail.com",
    description="Quantum Vacuum Engineering with Dark Photon Conversion",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tlcagford/stellaris-qed-engine",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)

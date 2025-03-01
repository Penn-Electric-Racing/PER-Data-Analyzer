from setuptools import setup, find_packages

setup(
    name="perda",  # Package name used for pip installation
    version="0.1.0",
    packages=find_packages(),  # Automatically finds `perda` package
    install_requires=[
        "numpy",
        "tqdm",
        "matplotlib"
    ],
    author="PER",
    description="A private data analysis package.",
    url="https://github.com/your-org/per-data-analyzer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)

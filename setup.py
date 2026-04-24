import os
from setuptools import setup, find_packages

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Dynamically read dependencies from requirements.txt
with open(os.path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = [
        line.strip() for line in f 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="literag",
    version="1.0.0",
    description="LiteRAG: Cost-Efficient Graph Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="LiteRAG Contributors", # Update after blind review
    url="https://github.com/macarronesc/literag",
    packages=find_packages(include=["literag", "literag.*"]),
    install_requires=requirements,
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="rag, graphrag, llm, knowledge-graph, retrieval-augmented-generation",
)
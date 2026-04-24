from setuptools import setup, find_packages

setup(
    name="literag",
    version="1.0.0",
    description="Cost-Efficient Graph Retrieval-Augmented Generation",
    packages=find_packages(include=["literag", "literag.*"]),
    # Note: Dependencies are managed in requirements.txt
)
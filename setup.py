from setuptools import setup, find_packages


setup(
    author="Megagon Labs, Tokyo.",
    author_email="vecscan@megagon.ai",
    description="vecscan: A Linear-scan-based High-speed Dense Vector Search Engine",
    entry_points={
        "console_scripts": [
            "vectorize = vecscan.vectorizer.__main__:main",
            "convert_to_safetensors = vecscan.vector_loader.__main__:main",
        ],
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.1",
        "numpy",
        "safetensors>=0.3.1",
        "tqdm",
    ],
    license="MIT",
    name="vecscan",
    packages=find_packages(include=["vecscan"]),
    url="https://github.com/megagonlabs/vecscan",
    version='2.0.0',
)

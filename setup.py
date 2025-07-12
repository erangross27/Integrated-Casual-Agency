"""
Setup script for ICA Framework
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ica-framework",
    version="0.1.0",
    author="ICA Development Team",
    author_email="ica@example.com",
    description="Integrated Causal Agency Framework for Artificial General Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/ica-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.4.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
        ],
        "viz": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "ica-demo=examples.demo:main",
            "ica-test=tests.test_components:run_all_tests",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

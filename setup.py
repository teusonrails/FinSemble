from setuptools import setup, find_packages

setup(
    name="finsemble",
    version="0.1.0",
    description="Framework de análise de texto financeiro baseado em classificadores Naive Bayes e meta-aprendizado",
    author="Mateus Ferreira",
    author_email="mateus.morais@aluno.unievangelica.edu.br",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "nltk",
        "spacy",
        "pgmpy",
        "pymc",
        "networkx",
        "matplotlib",
        "seaborn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
)
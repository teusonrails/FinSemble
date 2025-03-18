"""
Teste de verificação do ambiente de desenvolvimento.
Este teste garante que todas as bibliotecas principais estão instaladas corretamente.
"""

import pytest
import importlib


def test_import_core_libraries():
    """Testa a importação das bibliotecas principais."""
    libraries = [
        "numpy",
        "pandas",
        "sklearn",
        "nltk",
        "spacy",
        "pgmpy",
        "pymc",
        "networkx",
        "matplotlib",
        "seaborn",
    ]
    
    for lib in libraries:
        try:
            importlib.import_module(lib)
        except ImportError:
            pytest.fail(f"Falha ao importar a biblioteca: {lib}")


def test_import_project_modules():
    """Testa a importação dos módulos do projeto."""
    # Este teste será expandido à medida que módulos forem adicionados
    pass


def test_config_file_exists():
    """Verifica se o arquivo de configuração principal existe."""
    import os
    
    assert os.path.exists("config/config.yaml"), "Arquivo de configuração não encontrado"


def test_project_structure():
    """Verifica se a estrutura de diretórios do projeto está correta."""
    import os
    
    expected_dirs = [
        "config",
        "data",
        "data/raw",
        "data/processed",
        "data/knowledge_base",
        "src",
        "src/preprocessor",
        "src/classifiers",
        "src/aggregator",
        "src/contextualizer",
        "src/explainer",
        "src/utils",
        "tests",
        "tests/unit",
        "tests/integration",
        "tests/system",
        "notebooks",
        "docs",
    ]
    
    for directory in expected_dirs:
        assert os.path.isdir(directory), f"Diretório esperado não encontrado: {directory}"


if __name__ == "__main__":
    # Executa os testes diretamente se este arquivo for executado como script
    test_import_core_libraries()
    test_import_project_modules()
    test_config_file_exists()
    test_project_structure()
    
    print("Todos os testes de ambiente passaram com sucesso!")
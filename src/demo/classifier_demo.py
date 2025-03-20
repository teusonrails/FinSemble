"""
Script de demonstração do Classificador de Tipo/Categoria do FinSemble.

Este script demonstra a utilização do Classificador de Tipo/Categoria
para identificar diferentes tipos de documentos financeiros, incluindo
uma interface interativa aprimorada com tratamento robusto de erros.
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Adicionar o diretório raiz ao PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../..'))
sys.path.insert(0, project_root)

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.type_classifier import TypeClassifier

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_classificador")


def print_section(title: str) -> None:
    """
    Imprime um título de seção formatado.
    
    Args:
        title: Título da seção
    """
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, '='))
    print("=" * 80 + "\n")


def demo_classificacao_interativa(classifier: TypeClassifier, preprocessor: PreprocessorUniversal) -> None:
    """
    Demonstração interativa do classificador de tipo com validação robusta.
    
    Args:
        classifier: Classificador treinado
        preprocessor: Preprocessador inicializado
    """
    print_section("Demonstração Interativa")
    
    print("Digite alguns exemplos de textos financeiros para classificação (ou 'sair' para encerrar):")
    
    exemplos = [
        "A Empresa ABC S.A. (ABCD3) divulga seus resultados referentes ao 1º trimestre de 2023. A receita líquida atingiu R$ 1,2 bilhões, um aumento de 15% em relação ao mesmo período do ano anterior.",
        "FATO RELEVANTE: A Empresa XYZ S.A. (XYZ4), em cumprimento ao disposto na Instrução CVM nº 358/02, comunica aos seus acionistas e ao mercado em geral que aprovou plano de reestruturação operacional.",
        "A Administração da Empresa DEF S.A. (DEFS3) informa que o Conselho de Administração aprovou, em reunião realizada hoje, a distribuição de dividendos no valor de R$ 0,75 por ação.",
        "Análise técnica da ação ABCD3: o papel encontra-se próximo ao suporte em R$ 15,30, com médias móveis indicando tendência de alta. Indicadores técnicos sugerem possível reversão."
    ]
    
    print("\nExemplos sugeridos:")
    for i, exemplo in enumerate(exemplos):
        print(f"\n[{i+1}] {exemplo[:100]}...")
    
    # Loop de interação
    while True:
        try:
            print("\nDigite um número de exemplo (1-4) ou digite seu próprio texto (ou 'sair' para encerrar):")
            entrada = input("> ").strip()
            
            if entrada.lower() == 'sair':
                break
                
            if not entrada:
                print("Erro: O texto não pode estar vazio. Tente novamente.")
                continue
                
            # Selecionar exemplo ou usar entrada personalizada
            try:
                if entrada.isdigit() and 1 <= int(entrada) <= len(exemplos):
                    texto = exemplos[int(entrada) - 1]
                    print(f"\nTexto selecionado: {texto[:100]}...")
                else:
                    texto = entrada
            except ValueError:
                texto = entrada
            
            # Preprocessar texto com tratamento de exceções
            try:
                print("\nPreprocessando texto...")
                preprocessed = preprocessor.process(texto)
                if "error" in preprocessed:
                    print(f"Erro no preprocessamento: {preprocessed['error']}")
                    continue
            except Exception as e:
                print(f"Erro inesperado no preprocessamento: {str(e)}")
                continue
                
            # Classificar com tratamento de erros
            try:
                print("Classificando texto...")
                result = classifier.predict(preprocessed)
                if "error" in result:
                    print(f"Erro na classificação: {result['error']}")
                    continue
            except Exception as e:
                print(f"Erro inesperado na classificação: {str(e)}")
                continue
                
            # Mostrar resultado formatado
            print("\n" + "=" * 50)
            print("RESULTADO DA CLASSIFICAÇÃO")
            print("=" * 50)
            
            print(f"\nClasse: {result['readable_class']} ({result['predicted_class']})")
            print(f"Confiança: {result['confidence']:.2%}")
            
            print("\nDistribuição de probabilidades:")
            for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
                # Mostrar apenas classes com alguma probabilidade relevante
                if prob >= 0.01:
                    print(f"  {cls}: {prob:.2%}")
            
            if "relevant_terms" in result and result["relevant_terms"]:
                print("\nTermos relevantes para a classificação:")
                for term in result["relevant_terms"]:
                    print(f"  - {term}")
                    
            print(f"\nTempo de inferência: {result['inference_time']*1000:.1f} ms")
            
        except KeyboardInterrupt:
            print("\nOperação cancelada pelo usuário.")
            break
        except Exception as e:
            print(f"Erro inesperado: {str(e)}")
    
    print("\nDemonstração interativa encerrada")


def main():
    """Função principal para executar a demonstração."""
    print_section("Demonstração do Classificador de Tipo/Categoria")
    
    try:
        # Carregar configuração
        config = load_config()
        
        # Inicializar preprocessador
        preprocessor_config = get_config_section(config, "preprocessor")
        preprocessor = PreprocessorUniversal(preprocessor_config)
        
        # Verificar se existe modelo treinado
        model_path = "models/type_classifier.pkl"
        if os.path.exists(model_path):
            print(f"Modelo encontrado em {model_path}. Carregando...")
            
            # Inicializar e carregar o classificador
            classifier = TypeClassifier({"name": "type_classifier"})
            success = classifier.load_model(model_path)
            
            if success:
                print("Modelo carregado com sucesso")
                # Executar demonstração interativa
                demo_classificacao_interativa(classifier, preprocessor)
            else:
                print("Falha ao carregar o modelo. Verifique o arquivo e tente novamente.")
        else:
            print(f"Modelo não encontrado em {model_path}.")
            print("Execute o script de treinamento antes de usar a demonstração interativa.")
            
    except Exception as e:
        print(f"Erro ao iniciar a demonstração: {str(e)}")
    
    print_section("Demonstração Concluída")


if __name__ == "__main__":
    main()
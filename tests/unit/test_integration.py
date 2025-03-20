import sys
import os
import time
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.sentiment_classifier import SentimentAnalyzer
from src.classifiers.impact_modeler import ImpactModeler
from src.classifiers.type_classifier import TypeClassifier  # Supondo que exista
from src.aggregator.bayesian_aggregator import BayesianAggregator

def initialize_components(config):
    """Inicializa todos os componentes do sistema."""
    print("Inicializando componentes do sistema...")
    
    # Extrair configurações específicas
    preprocessor_config = get_config_section(config, "preprocessor")
    classifier_configs = get_config_section(config, "classifiers")
    aggregator_config = get_config_section(config, "aggregator")
    
    # Inicializar componentes
    preprocessor = PreprocessorUniversal(preprocessor_config)
    type_classifier = TypeClassifier(classifier_configs["type_classifier"])
    sentiment_analyzer = SentimentAnalyzer(classifier_configs["sentiment_analyzer"])
    impact_modeler = ImpactModeler(classifier_configs["impact_modeler"])
    aggregator = BayesianAggregator(aggregator_config)
    
    return {
        "preprocessor": preprocessor,
        "type_classifier": type_classifier,
        "sentiment_analyzer": sentiment_analyzer,
        "impact_modeler": impact_modeler,
        "aggregator": aggregator
    }

def train_classifiers(components, train_data=None):
    """Treina os classificadores com dados de exemplo."""
    print("Treinando classificadores...")
    
    # Obter componentes
    preprocessor = components["preprocessor"]
    type_classifier = components["type_classifier"]
    sentiment_analyzer = components["sentiment_analyzer"]
    impact_modeler = components["impact_modeler"]
    
    # Criar dados de treinamento simples se não fornecidos
    if train_data is None:
        train_data = {
            "type": {
                "texts": [
                    "RELATÓRIO TRIMESTRAL: Resultados financeiros do terceiro trimestre.",
                    "RELATÓRIO ANUAL: Demonstrativos financeiros auditados.",
                    "FATO RELEVANTE: Comunicamos a aquisição da empresa XYZ.",
                    "COMUNICADO AO MERCADO: Esclarecimentos sobre notícias.",
                    "GUIDANCE: Projeções para o próximo exercício fiscal."
                ],
                "labels": ["report", "report", "announcement", "announcement", "guidance"]
            },
            "sentiment": {
                "texts": [
                    "A empresa registrou crescimento recorde na receita.",
                    "Os resultados superaram as expectativas dos analistas.",
                    "A companhia reportou prejuízo no trimestre.",
                    "Os indicadores financeiros deterioraram significativamente.",
                    "A empresa mantém suas projeções para o ano corrente."
                ],
                "labels": ["positive", "positive", "negative", "negative", "neutral"]
            },
            "impact": {
                "texts": [
                    "A decisão terá impacto significativo nos próximos anos.",
                    "A aquisição transformará completamente o setor.",
                    "O novo produto deve ter impacto moderado nos resultados.",
                    "A iniciativa trará contribuição média para o crescimento.",
                    "A mudança terá efeito limitado nas operações."
                ],
                "labels": ["high", "high", "medium", "medium", "low"]
            }
        }
    
    results = {}
    
    # Treinar classificador de tipo
    type_data = []
    for text in train_data["type"]["texts"]:
        preprocessed = preprocessor.process(text)
        type_data.append(preprocessed)
    
    type_result = type_classifier.train(type_data, train_data["type"]["labels"])
    results["type"] = type_result["success"] if "success" in type_result else False
    
    # Treinar analisador de sentimento
    sentiment_data = []
    for text in train_data["sentiment"]["texts"]:
        preprocessed = preprocessor.process(text)
        sentiment_data.append(preprocessed)
    
    sentiment_result = sentiment_analyzer.train(sentiment_data, train_data["sentiment"]["labels"])
    results["sentiment"] = sentiment_result["success"] if "success" in sentiment_result else False
    
    # Treinar modelador de impacto
    impact_data = []
    for text in train_data["impact"]["texts"]:
        preprocessed = preprocessor.process(text)
        impact_data.append(preprocessed)
    
    impact_result = impact_modeler.train(impact_data, train_data["impact"]["labels"])
    results["impact"] = impact_result["success"] if "success" in impact_result else False
    
    # Verificar se todos os treinamentos foram bem-sucedidos
    all_success = all(results.values())
    if all_success:
        print("Todos os classificadores treinados com sucesso!")
    else:
        print("Alguns classificadores não foram treinados corretamente:")
        for classifier, success in results.items():
            print(f"  {classifier}: {'Sucesso' if success else 'Falha'}")
    
    return all_success

def process_text(components, text):
    """Processa um texto completo pelo pipeline do FinSemble."""
    print(f"\nProcessando texto: '{text[:100]}...'")
    
    start_time = time.time()
    
    # Obter componentes
    preprocessor = components["preprocessor"]
    type_classifier = components["type_classifier"]
    sentiment_analyzer = components["sentiment_analyzer"]
    impact_modeler = components["impact_modeler"]
    aggregator = components["aggregator"]
    
    # Etapa 1: Preprocessamento
    print("\n1. Preprocessamento do texto...")
    preprocessed = preprocessor.process(text)
    if "error" in preprocessed:
        print(f"Erro no preprocessamento: {preprocessed['error']}")
        return None
    
    preprocessing_time = time.time() - start_time
    print(f"Texto normalizado: '{preprocessed['normalized_text'][:100]}...'")
    print(f"Tempo de preprocessamento: {preprocessing_time:.3f}s")
    
    # Etapa 2: Classificadores especializados
    print("\n2. Execução dos classificadores especializados...")
    classifier_start = time.time()
    
    # Classificador de tipo
    type_result = type_classifier.predict(preprocessed)
    if "error" in type_result:
        print(f"Erro no classificador de tipo: {type_result['error']}")
        type_result = {"predicted_class": "unknown", "confidence": 0.0, "probabilities": {}}
    
    # Analisador de sentimento
    sentiment_result = sentiment_analyzer.predict(preprocessed)
    if "error" in sentiment_result:
        print(f"Erro no analisador de sentimento: {sentiment_result['error']}")
        sentiment_result = {"predicted_class": "neutral", "confidence": 0.0, "probabilities": {}}
    
    # Modelador de impacto
    impact_result = impact_modeler.predict(preprocessed)
    if "error" in impact_result:
        print(f"Erro no modelador de impacto: {impact_result['error']}")
        impact_result = {"predicted_class": "medium", "confidence": 0.0, "probabilities": {}}
    
    classifier_time = time.time() - classifier_start
    
    # Mostrar resultados dos classificadores
    print(f"\nTipo: {type_result.get('readable_class', 'Desconhecido')} "
          f"(confiança: {type_result.get('confidence', 0):.2%})")
    print(f"Sentimento: {sentiment_result.get('readable_class', 'Neutro')} "
          f"(confiança: {sentiment_result.get('confidence', 0):.2%})")
    print(f"Impacto: {impact_result.get('readable_class', 'Médio')} "
          f"(confiança: {impact_result.get('confidence', 0):.2%})")
    print(f"Tempo de classificação: {classifier_time:.3f}s")
    
    # Etapa 3: Agregação
    print("\n3. Agregação dos resultados...")
    aggregation_start = time.time()
    
    # Organizar resultados dos classificadores
    classifier_outputs = {
        "type_classifier": type_result,
        "sentiment_analyzer": sentiment_result,
        "impact_modeler": impact_result
    }
    
    # Fazer agregação
    aggregation_result = aggregator.aggregate(classifier_outputs)
    
    aggregation_time = time.time() - aggregation_start
    
    # Verificar erro
    if "error" in aggregation_result:
        print(f"Erro na agregação: {aggregation_result['error']}")
        return None
    
    # Mostrar resultado final
    print("\nResultado Final da Análise:")
    if "market_direction" in aggregation_result:
        print(f"Direção de Mercado: {aggregation_result['market_direction']} "
              f"(confiança: {aggregation_result['market_direction_confidence']:.2%})")
    
    if "investment_horizon" in aggregation_result:
        print(f"Horizonte de Investimento: {aggregation_result['investment_horizon']} "
              f"(confiança: {aggregation_result['investment_horizon_confidence']:.2%})")
    
    if "risk_level" in aggregation_result:
        print(f"Nível de Risco: {aggregation_result['risk_level']} "
              f"(confiança: {aggregation_result['risk_level_confidence']:.2%})")
    
    if "recommendation" in aggregation_result:
        print(f"\nRecomendação: {aggregation_result['recommendation']}")
    
    # Explicações
    if "explanations" in aggregation_result:
        print("\nExplicação:")
        print(aggregation_result["explanations"]["summary"])
    
    print(f"Tempo de agregação: {aggregation_time:.3f}s")
    
    # Tempo total
    total_time = time.time() - start_time
    print(f"\nTempo total de processamento: {total_time:.3f}s")
    
    return aggregation_result

# Função principal
def main():
    # Carregar configuração
    config = load_config()
    
    # Inicializar componentes
    components = initialize_components(config)
    
    # Treinar classificadores
    if not train_classifiers(components):
        print("Falha no treinamento de alguns classificadores. Teste interrompido.")
        return
    
    # Exemplos de textos financeiros
    exemplos = [
        """
        RELATÓRIO TRIMESTRAL: A Empresa XYZ S.A. (XYZS3) apresenta seus resultados 
        referentes ao primeiro trimestre de 2023. A receita líquida totalizou 
        R$ 1,2 bilhão, um aumento de 15% em relação ao mesmo período do ano anterior. 
        O EBITDA ajustado foi de R$ 350 milhões, com margem de 29,2%, representando 
        um crescimento de 18% ano contra ano. O lucro líquido atingiu R$ 180 milhões, 
        superando as expectativas do mercado. A companhia mantém suas projeções de 
        crescimento para o ano de 2023 entre 12% e 15%.
        """,
        
        """
        FATO RELEVANTE: A Empresa ABC S.A. (ABCD3) comunica aos seus acionistas e ao 
        mercado em geral que concluiu nesta data a aquisição da XYZ Tecnologia Ltda., 
        pelo valor de R$ 500 milhões. A transação representa um múltiplo de 12x EBITDA 
        e deve impactar negativamente o resultado do próximo trimestre em razão dos 
        custos de integração. A Companhia espera capturar sinergias anuais de 
        aproximadamente R$ 50 milhões a partir do segundo ano após a conclusão.
        """,
        
        """
        COMUNICADO AO MERCADO: A Empresa DEF S.A. (DEFS3) informa que mantém suas 
        projeções (guidance) para o ano de 2023, com crescimento esperado entre 3% e 5% 
        na receita líquida e margem EBITDA estável entre 22% e 24%, em linha com o 
        comunicado ao mercado divulgado em 15 de fevereiro de 2023. A Companhia reafirma 
        seu compromisso com a disciplina financeira e geração de valor aos acionistas.
        """
    ]
    
    # Processar cada exemplo
    for i, exemplo in enumerate(exemplos):
        print(f"\n{'='*80}")
        print(f" EXEMPLO {i+1} ".center(80, '='))
        print(f"{'='*80}")
        
        result = process_text(components, exemplo)
        
        if i < len(exemplos) - 1:
            input("\nPressione Enter para continuar para o próximo exemplo...")

if __name__ == "__main__":
    main()
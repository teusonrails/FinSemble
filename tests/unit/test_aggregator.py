import sys
import os
import json
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.aggregator.bayesian_aggregator import BayesianAggregator

# Carregar configurações
config = load_config()
aggregator_config = get_config_section(config, "aggregator")

# Inicializar agregador
aggregator = BayesianAggregator(aggregator_config)

# Simular saídas dos classificadores
def simulate_classifier_outputs(scenario="positive_high"):
    """Simula diferentes cenários de saída dos classificadores."""
    
    if scenario == "positive_high":
        return {
            "type_classifier": {
                "predicted_class": "report",
                "readable_class": "Relatório",
                "confidence": 0.85,
                "probabilities": {
                    "report": 0.85,
                    "announcement": 0.10,
                    "guidance": 0.05
                }
            },
            "sentiment_analyzer": {
                "predicted_class": "positive",
                "readable_class": "Positivo",
                "confidence": 0.92,
                "probabilities": {
                    "positive": 0.92,
                    "neutral": 0.06,
                    "negative": 0.02
                }
            },
            "impact_modeler": {
                "predicted_class": "high",
                "readable_class": "Alto Impacto",
                "confidence": 0.78,
                "probabilities": {
                    "high": 0.78,
                    "medium": 0.20,
                    "low": 0.02
                }
            }
        }
    elif scenario == "negative_medium":
        return {
            "type_classifier": {
                "predicted_class": "announcement",
                "readable_class": "Comunicado",
                "confidence": 0.76,
                "probabilities": {
                    "announcement": 0.76,
                    "report": 0.15,
                    "guidance": 0.09
                }
            },
            "sentiment_analyzer": {
                "predicted_class": "negative",
                "readable_class": "Negativo",
                "confidence": 0.83,
                "probabilities": {
                    "negative": 0.83,
                    "neutral": 0.12,
                    "positive": 0.05
                }
            },
            "impact_modeler": {
                "predicted_class": "medium",
                "readable_class": "Médio Impacto",
                "confidence": 0.65,
                "probabilities": {
                    "medium": 0.65,
                    "high": 0.25,
                    "low": 0.10
                }
            }
        }
    elif scenario == "neutral_low":
        return {
            "type_classifier": {
                "predicted_class": "guidance",
                "readable_class": "Guidance",
                "confidence": 0.70,
                "probabilities": {
                    "guidance": 0.70,
                    "report": 0.20,
                    "announcement": 0.10
                }
            },
            "sentiment_analyzer": {
                "predicted_class": "neutral",
                "readable_class": "Neutro",
                "confidence": 0.75,
                "probabilities": {
                    "neutral": 0.75,
                    "positive": 0.15,
                    "negative": 0.10
                }
            },
            "impact_modeler": {
                "predicted_class": "low",
                "readable_class": "Baixo Impacto",
                "confidence": 0.68,
                "probabilities": {
                    "low": 0.68,
                    "medium": 0.27,
                    "high": 0.05
                }
            }
        }
    else:
        return {
            "type_classifier": {
                "predicted_class": "unknown",
                "readable_class": "Desconhecido",
                "confidence": 0.40,
                "probabilities": {
                    "report": 0.40,
                    "announcement": 0.35,
                    "guidance": 0.25
                }
            },
            "sentiment_analyzer": {
                "predicted_class": "neutral",
                "readable_class": "Neutro",
                "confidence": 0.45,
                "probabilities": {
                    "neutral": 0.45,
                    "positive": 0.30,
                    "negative": 0.25
                }
            },
            "impact_modeler": {
                "predicted_class": "medium",
                "readable_class": "Médio Impacto",
                "confidence": 0.50,
                "probabilities": {
                    "medium": 0.50,
                    "high": 0.30,
                    "low": 0.20
                }
            }
        }

# Testar diferentes cenários
scenarios = ["positive_high", "negative_medium", "neutral_low", "ambiguous"]

for scenario in scenarios:
    print(f"\n{'='*80}")
    print(f" Testando cenário: {scenario} ".center(80, '='))
    print(f"{'='*80}")
    
    # Obter saídas simuladas dos classificadores
    classifier_outputs = simulate_classifier_outputs(scenario)
    
    # Mostrar entradas
    print("\nEntradas dos classificadores:")
    for classifier, output in classifier_outputs.items():
        print(f"\n{classifier}:")
        print(f"  Classe: {output['readable_class']} (confiança: {output['confidence']:.2%})")
    
    # Executar agregação
    result = aggregator.aggregate(classifier_outputs)
    
    # Verificar erro
    if "error" in result:
        print(f"\nErro na agregação: {result['error']}")
        continue
    
    # Mostrar resultado da agregação
    print("\nResultado da Agregação Bayesiana:")
    
    if "market_direction" in result:
        print(f"Direção de Mercado: {result['market_direction']} (confiança: {result['market_direction_confidence']:.2%})")
    
    if "investment_horizon" in result:
        print(f"Horizonte de Investimento: {result['investment_horizon']} (confiança: {result['investment_horizon_confidence']:.2%})")
    
    if "risk_level" in result:
        print(f"Nível de Risco: {result['risk_level']} (confiança: {result['risk_level_confidence']:.2%})")
    
    if "recommendation" in result:
        print(f"\nRecomendação: {result['recommendation']}")
    
    # Explicações
    if "explanations" in result:
        print("\nExplicação:")
        print(result["explanations"]["summary"])
        
        print("\nFatores Contribuintes:")
        for factor in result["explanations"]["factors"][:3]:  # Limitando a 3 fatores
            print(f"  - {factor['description']}")
    
    # Método de inferência
    if "inference_method" in result:
        print(f"\nMétodo de inferência utilizado: {result['inference_method']}")
    
    print(f"\nTempo de processamento: {result.get('inference_time', 0):.3f}s")
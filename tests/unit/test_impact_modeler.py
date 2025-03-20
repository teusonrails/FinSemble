import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.impact_modeler import ImpactModeler

# Carregar configurações
config = load_config()
preprocessor_config = get_config_section(config, "preprocessor")
impact_config = get_config_section(config, "classifiers")["impact_modeler"]

# Inicializar componentes
preprocessor = PreprocessorUniversal(preprocessor_config)
impact_modeler = ImpactModeler(impact_config)

# Exemplos de textos
exemplos = [
    "A Empresa XYZ anunciou um aumento significativo de 35% nos investimentos em P&D para os próximos 5 anos.",
    "A empresa reportou uma pequena redução de 3% na margem bruta, sem impacto expressivo nos resultados.",
    "O conselho decidiu manter os níveis atuais de distribuição de dividendos para o próximo trimestre."
]

# Iniciar treinamento
print("Treinando o modelador de impacto...")
training_data = []
labels = []

# Dados de treinamento simples
train_texts = [
    # Alto impacto
    "A empresa anunciou uma aquisição massiva avaliada em R$ 2 bilhões que transformará o setor.",
    "Os resultados mostram um crescimento extraordinário de 40% na receita anual.",
    "A companhia reportou uma perda significativa de R$ 500 milhões, o pior resultado de sua história.",
    "A decisão regulatória terá impacto profundo nas operações por vários anos.",
    "A parceria estratégica abrirá mercados globais com potencial enorme para a empresa.",
    # Médio impacto
    "A empresa reportou um aumento moderado de 12% na receita trimestral.",
    "Os custos operacionais apresentaram uma redução razoável de 8% no período.",
    "A nova linha de produtos deve contribuir de forma considerável para o crescimento no próximo ano.",
    "O investimento de médio prazo trará resultados graduais para a empresa.",
    "A reestruturação terá um impacto notável, mas gerenciável, nas operações.",
    # Baixo impacto
    "A empresa fez pequenos ajustes em sua política de preços.",
    "A flutuação cambial teve efeito limitado nas operações internacionais.",
    "A mudança na liderança regional terá impacto mínimo nos resultados de curto prazo.",
    "O novo sistema operacional apresenta melhorias sutis na eficiência.",
    "A atualização da marca representa uma mudança discreta na estratégia de marketing."
]

train_labels = ["high", "high", "high", "high", "high",
               "medium", "medium", "medium", "medium", "medium",
               "low", "low", "low", "low", "low"]

# Preprocessar dados de treinamento
for text, label in zip(train_texts, train_labels):
    preprocessed = preprocessor.process(text)
    training_data.append(preprocessed)
    labels.append(label)

# Treinar o modelo
training_result = impact_modeler.train(training_data, labels, validation_split=0.2)
print("Treinamento concluído.")

if "success" in training_result and training_result["success"]:
    print("Treinamento bem-sucedido!")
    if "metrics" in training_result:
        print("\nMétricas na validação:")
        for metric, value in training_result["metrics"].items():
            print(f"  {metric}: {value:.4f}")
else:
    print(f"Erro no treinamento: {training_result.get('error', 'Erro desconhecido')}")
    sys.exit(1)

# Testar o modelo
print("\nTestando o modelador de impacto com exemplos:")
for i, texto in enumerate(exemplos):
    print(f"\nExemplo {i+1}:")
    print(texto)
    
    # Preprocessar
    preprocessed = preprocessor.process(texto)
    
    # Analisar impacto
    result = impact_modeler.predict(preprocessed)
    
    # Mostrar resultado
    print("\nResultado:")
    print(f"Impacto: {result['readable_class']} (confiança: {result['confidence']:.2%})")
    print("\nProbabilidades:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.2%}")
    
    # Características importantes
    if "important_features" in result:
        print("\nCaracterísticas importantes:")
        for feature, value in result["important_features"]:
            print(f"  {feature}: {value:.4f}")
    
    # Explicação
    explanation = impact_modeler.get_impact_explanation(preprocessed)
    print("\nExplicação:")
    print(explanation["summary"])
    
    if "contributing_factors" in explanation:
        print("\nFatores contribuintes:")
        for factor in explanation["contributing_factors"][:2]:  # Limitando a 2 fatores
            print(f"  {factor['description']}")
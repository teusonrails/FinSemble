import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.type_classifier import TypeClassifier  # Supondo que exista esta classe

# Carregar configurações
config = load_config()
preprocessor_config = get_config_section(config, "preprocessor")
type_config = get_config_section(config, "classifiers")["type_classifier"]

# Inicializar componentes
preprocessor = PreprocessorUniversal(preprocessor_config)
type_classifier = TypeClassifier(type_config)

# Exemplos de textos
exemplos = [
    "RELATÓRIO TRIMESTRAL: A Empresa XYZ S.A. apresenta seu relatório do 1º trimestre de 2023...",
    "FATO RELEVANTE: A Empresa ABC comunica aos seus acionistas e ao mercado em geral...",
    "GUIDANCE: A administração da Empresa XYZ projeta um crescimento de 10-15% para o próximo ano..."
]

# Iniciar treinamento
print("Treinando o classificador de tipo...")
training_data = []
labels = []

# Dados de treinamento simples
train_texts = [
    # Relatórios
    "RELATÓRIO TRIMESTRAL: Apresentamos os resultados do terceiro trimestre de 2023.",
    "RELATÓRIO ANUAL: Demonstrações financeiras auditadas do exercício encerrado em 31 de dezembro.",
    "RELATÓRIO DE ADMINISTRAÇÃO: O Conselho apresenta as realizações do período.",
    "RELATÓRIO OPERACIONAL: Detalhes sobre as operações da companhia no período.",
    "RELATÓRIO DE SUSTENTABILIDADE: Métricas ESG da empresa no último ano fiscal.",
    # Anúncios
    "FATO RELEVANTE: A empresa comunica a aquisição da companhia XYZ por R$ 500 milhões.",
    "COMUNICADO AO MERCADO: Esclarecimentos sobre notícias veiculadas na mídia.",
    "AVISO AOS ACIONISTAS: Pagamento de dividendos aprovado pelo Conselho.",
    "COMUNICADO: Mudanças na estrutura organizacional da companhia.",
    "ANÚNCIO: Nova parceria estratégica firmada com empresa internacional.",
    # Guidance
    "GUIDANCE: Projeções de crescimento para o próximo exercício fiscal.",
    "PERSPECTIVAS: A administração espera um aumento de 10-15% na receita.",
    "PROJEÇÕES: Estimativas de resultados para os próximos trimestres.",
    "ESTIMATIVAS: A companhia projeta margem EBITDA entre 25-30% para 2023.",
    "GUIDANCE CORPORATIVO: Metas financeiras e operacionais para o próximo triênio."
]

train_labels = ["report", "report", "report", "report", "report",
               "announcement", "announcement", "announcement", "announcement", "announcement",
               "guidance", "guidance", "guidance", "guidance", "guidance"]

# Preprocessar dados de treinamento
for text, label in zip(train_texts, train_labels):
    preprocessed = preprocessor.process(text)
    training_data.append(preprocessed)
    labels.append(label)

# Treinar o modelo
training_result = type_classifier.train(training_data, labels, validation_split=0.2)
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
print("\nTestando o classificador de tipo com exemplos:")
for i, texto in enumerate(exemplos):
    print(f"\nExemplo {i+1}:")
    print(texto)
    
    # Preprocessar
    preprocessed = preprocessor.process(texto)
    
    # Classificar tipo
    result = type_classifier.predict(preprocessed)
    
    # Mostrar resultado
    print("\nResultado:")
    print(f"Tipo: {result['readable_class']} (confiança: {result['confidence']:.2%})")
    print("\nProbabilidades:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.2%}")
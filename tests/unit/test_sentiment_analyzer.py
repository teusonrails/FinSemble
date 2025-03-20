import sys
import os
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.sentiment_classifier import SentimentAnalyzer

# Carregar configurações
config = load_config()
preprocessor_config = get_config_section(config, "preprocessor")
sentiment_config = get_config_section(config, "classifiers")["sentiment_analyzer"]

# Inicializar componentes
preprocessor = PreprocessorUniversal(preprocessor_config)
sentiment_analyzer = SentimentAnalyzer(sentiment_config)

# Exemplos de textos
exemplos = [
    "A Empresa XYZ registrou um crescimento de 20% na receita líquida, superando as expectativas dos analistas.",
    "A empresa reportou um prejuízo líquido de R$ 150 milhões, pior que as estimativas do mercado.",
    "A empresa divulga seus resultados do trimestre, com receita líquida de R$ 1,2 bilhão, em linha com as expectativas."
]

# Iniciar treinamento
print("Treinando o analisador de sentimento...")
training_data = []
labels = []

# Dados de treinamento simples
train_texts = [
    # Positivos
    "A empresa registrou lucro recorde neste trimestre.",
    "As ações subiram 15% após o anúncio dos resultados.",
    "A aquisição foi finalizada com sucesso.",
    "Os indicadores financeiros mostram forte recuperação.",
    "O conselho aprovou aumento nos dividendos.",
    # Negativos
    "A empresa reportou prejuízo pelo terceiro trimestre consecutivo.",
    "As ações caíram 10% após o anúncio.",
    "A dívida da empresa aumentou significativamente.",
    "A empresa perdeu participação de mercado para concorrentes.",
    "A receita ficou abaixo das estimativas dos analistas.",
    # Neutros
    "A empresa realizará sua assembleia no próximo mês.",
    "O relatório trimestral será divulgado na próxima semana.",
    "O conselho se reunirá para discutir os próximos passos.",
    "A empresa mantém suas projeções para o ano.",
    "O mercado aguarda o posicionamento da empresa."
]

train_labels = ["positive", "positive", "positive", "positive", "positive",
               "negative", "negative", "negative", "negative", "negative",
               "neutral", "neutral", "neutral", "neutral", "neutral"]

# Preprocessar dados de treinamento
for text, label in zip(train_texts, train_labels):
    preprocessed = preprocessor.process(text)
    training_data.append(preprocessed)
    labels.append(label)

# Treinar o modelo
training_result = sentiment_analyzer.train(training_data, labels, validation_split=0.2)
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
print("\nTestando o analisador de sentimento com exemplos:")
for i, texto in enumerate(exemplos):
    print(f"\nExemplo {i+1}:")
    print(texto)
    
    # Preprocessar
    preprocessed = preprocessor.process(texto)
    
    # Analisar sentimento
    result = sentiment_analyzer.predict(preprocessed)
    
    # Mostrar resultado
    print("\nResultado:")
    print(f"Sentimento: {result['readable_class']} (confiança: {result['confidence']:.2%})")
    print("\nProbabilidades:")
    for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {prob:.2%}")
    
    # Explicação
    explanation = sentiment_analyzer.get_sentiment_explanation(preprocessed)
    print("\nExplicação:")
    print(explanation["summary"])
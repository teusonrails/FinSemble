"""
Script de demonstração do Analisador de Sentimento do FinSemble.

Este script demonstra o uso do Analisador de Sentimento baseado em Complement Naive Bayes
para classificar textos financeiros com base no sentimento (positivo, negativo, neutro).
"""

import os
import sys
import time
import json
import logging
from pprint import pprint

# Adicionar o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.classifiers.sentiment_classifier import SentimentAnalyzer
from src.utils.lexicon_manager import LexiconManager

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_sentiment_analyzer")

def print_section(title):
    """Imprime um título de seção formatado."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, '='))
    print("=" * 80 + "\n")

def print_result(result, show_details=True):
    """Imprime o resultado da análise de sentimento de forma formatada."""
    if "error" in result:
        print(f"ERRO: {result['error']}")
        return
        
    sentiment = result["readable_class"]
    confidence = result["confidence"]
    
    # Usar cores para diferentes sentimentos (ANSI escape codes)
    if sentiment == "Positivo":
        sentiment_colored = f"\033[92m{sentiment}\033[0m"  # Verde
    elif sentiment == "Negativo":
        sentiment_colored = f"\033[91m{sentiment}\033[0m"  # Vermelho
    else:
        sentiment_colored = f"\033[93m{sentiment}\033[0m"  # Amarelo
        
    print(f"Sentimento: {sentiment_colored} (confiança: {confidence:.2%})")
    
    if show_details:
        # Mostrar probabilidades para cada classe
        print("\nDistribuição de probabilidades:")
        for cls, prob in sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True):
            class_name = result["class_mapping"][cls] if cls in result.get("class_mapping", {}) else cls
            print(f" {class_name}: {prob:.2%}")
            
        # Mostrar termos relevantes
        if "relevant_terms" in result and result["relevant_terms"]:
            print("\nTermos relevantes para a classificação:")
            for term in result["relevant_terms"]:
                print(f" - {term}")
                
        # Mostrar tempo de inferência
        if "inference_time" in result:
            print(f"\nTempo de processamento: {result['inference_time']*1000:.2f} ms")

def create_sentiment_analyzer():
    """Cria e configura o analisador de sentimento."""
    print_section("Inicialização do Analisador de Sentimento")
    
    try:
        # Carregar configuração
        config = load_config()
        sentiment_config = get_config_section(config, "sentiment_analyzer")
        
        # Se não existir configuração específica, criar uma básica
        if not sentiment_config:
            sentiment_config = {
                "name": "sentiment_analyzer",
                "alpha": 1.0,
                "fit_prior": True,
                "norm": True,
                "lexicon_path": "data/lexicons/financial_lexicon.json",
                "feature_extraction": {
                    "max_features": 3000,
                    "min_df": 2,
                    "ngram_range": (1, 3),
                    "use_idf": True
                },
                "class_mapping": {
                    "positive": "Positivo",
                    "negative": "Negativo",
                    "neutral": "Neutro"
                }
            }
            
        # Verificar existência do léxico e criar se necessário
        lexicon_path = sentiment_config.get("lexicon_path")
        if lexicon_path and not os.path.exists(lexicon_path):
            os.makedirs(os.path.dirname(lexicon_path), exist_ok=True)
            lexicon_manager = LexiconManager()
            lexicon_manager.save_lexicon(lexicon_path)
            print(f"Léxico financeiro básico criado em: {lexicon_path}")
            
        # Inicializar o analisador
        analyzer = SentimentAnalyzer(sentiment_config)
        
        print(f"Analisador de Sentimento v{analyzer.version} inicializado com sucesso:")
        print(f" - Algoritmo: Complement Naive Bayes com normalização de peso")
        print(f" - Alpha: {analyzer.alpha}")
        print(f" - Normalização: {'Ativada' if analyzer.norm else 'Desativada'}")
        print(f" - Máximo de features: {analyzer.max_features}")
        print(f" - Classes: {', '.join(analyzer.class_mapping.values())}")
        print(f" - Léxico: {sum(len(terms) for terms in analyzer.sentiment_lexicon.values())} termos")
        
        return analyzer
        
    except Exception as e:
        logger.error(f"Erro ao inicializar o Analisador de Sentimento: {str(e)}")
        raise

def create_preprocessor():
    """Cria e configura o preprocessador universal."""
    try:
        # Carregar configuração
        config = load_config()
        preprocessor_config = get_config_section(config, "preprocessor")
        
        # Inicializar o preprocessador
        preprocessor = PreprocessorUniversal(preprocessor_config)
        print("Preprocessador Universal inicializado com sucesso.")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Erro ao inicializar o Preprocessador Universal: {str(e)}")
        raise

def train_sentiment_analyzer(analyzer, preprocessor):
    """Treina o analisador de sentimento com exemplos financeiros."""
    print_section("Treinamento do Analisador de Sentimento")
    
    # Dados de treinamento: textos financeiros com sentimentos anotados
    training_data = [
        # Exemplos Positivos
        {
            "text": "A Empresa XYZ S.A. registrou um crescimento de 20% na receita líquida do primeiro trimestre de 2023, superando as expectativas dos analistas. O lucro líquido aumentou 15% em relação ao mesmo período do ano anterior, refletindo a estratégia de expansão bem-sucedida.",
            "label": "positive"
        },
        {
            "text": "Comunicado ao Mercado: A Empresa ABC anuncia a conclusão bem-sucedida da aquisição da Empresa DEF, fortalecendo sua posição de liderança no setor. Esta aquisição estratégica deverá contribuir positivamente para os resultados já no próximo trimestre.",
            "label": "positive"
        },
        {
            "text": "O Conselho de Administração aprovou a distribuição de dividendos extraordinários no valor de R$ 2,50 por ação, refletindo o desempenho excepcional e a sólida geração de caixa da companhia nos últimos meses.",
            "label": "positive"
        },
        {
            "text": "As ações da empresa valorizaram 8% após o anúncio da nova parceria estratégica com um grande player internacional, que deve abrir novos mercados e impulsionar o crescimento nos próximos anos.",
            "label": "positive"
        },
        {
            "text": "A empresa conseguiu reduzir significativamente seu endividamento através de uma gestão eficiente de capital de giro e forte geração de caixa operacional, melhorando seus indicadores financeiros.",
            "label": "positive"
        },
        # Exemplos Negativos
        {
            "text": "A Empresa XYZ S.A. reportou um prejuízo líquido de R$ 150 milhões no primeiro trimestre de 2023, pior que as estimativas do mercado. A margem EBITDA caiu de 15% para 8%, refletindo o aumento de custos e a forte pressão competitiva.",
            "label": "negative"
        },
        {
            "text": "Fato Relevante: A Empresa ABC informa que teve sua classificação de risco rebaixada pela agência Moody's devido ao aumento do endividamento e deterioração dos indicadores de liquidez nos últimos dois trimestres.",
            "label": "negative"
        },
        {
            "text": "A empresa anunciou o adiamento do pagamento de dividendos previstos para este trimestre em função da necessidade de preservação de caixa diante do cenário econômico desafiador.",
            "label": "negative"
        },
        {
            "text": "As ações da companhia caíram 12% após a divulgação dos resultados trimestrais abaixo do esperado e do guidance reduzido para o restante do ano fiscal.",
            "label": "negative"
        },
        {
            "text": "A empresa enfrenta processo de investigação por possíveis irregularidades contábeis, o que pode resultar em multas significativas e impactar negativamente sua reputação no mercado.",
            "label": "negative"
        },
        # Exemplos Neutros
        {
            "text": "A Empresa XYZ S.A. divulga seus resultados do primeiro trimestre de 2023, com receita líquida de R$ 1,2 bilhão e EBITDA de R$ 300 milhões, em linha com as expectativas do mercado.",
            "label": "neutral"
        },
        {
            "text": "Comunicado: A Empresa ABC informa que realizará sua Assembleia Geral Ordinária no dia 30 de abril para deliberar sobre as demonstrações financeiras e a eleição do Conselho de Administração.",
            "label": "neutral"
        },
        {
            "text": "A empresa mantém seu guidance para o ano de 2023, com projeção de crescimento entre 3% e 5% na receita líquida e margem EBITDA estável entre 20% e 22%.",
            "label": "neutral"
        },
        {
            "text": "O Conselho de Administração aprovou a renovação do programa de recompra de ações nos mesmos moldes do programa anterior, sem alterações significativas nas condições.",
            "label": "neutral"
        },
        {
            "text": "A empresa concluiu a rolagem de sua dívida sem mudanças relevantes nas taxas ou prazos, mantendo seu perfil de endividamento atual conforme esperado pelo mercado.",
            "label": "neutral"
        }
    ]
    
    print(f"Preparando {len(training_data)} exemplos para treinamento...")
    
    # Preprocessar os dados de treinamento
    preprocessed_data = []
    labels = []
    
    for item in training_data:
        # Preprocessar o texto
        preprocessed = preprocessor.process(item["text"])
        preprocessed_data.append(preprocessed)
        labels.append(item["label"])
        
    print("Iniciando treinamento do modelo...")
    start_time = time.time()
    
    # Treinar o modelo
    training_result = analyzer.train(preprocessed_data, labels, validation_split=0.2)
    training_time = time.time() - start_time
    
    print(f"Treinamento concluído em {training_time:.2f} segundos.")
    
    # Mostrar resultados do treinamento
    if "success" in training_result and training_result["success"]:
        print("\nTreinamento bem-sucedido!")
        print(f"Número de amostras: {training_result.get('num_samples', len(training_data))}")
        print(f"Número de características: {training_result.get('num_features', 'N/A')}")
        
        # Mostrar métricas de validação, se disponíveis
        if "metrics" in training_result:
            metrics = training_result["metrics"]
            print("\nMétricas na validação:")
            print(f" Acurácia: {metrics.get('accuracy', 0):.4f}")
            print(f" Precisão: {metrics.get('precision', 0):.4f}")
            print(f" Recall: {metrics.get('recall', 0):.4f}")
            print(f" F1-Score: {metrics.get('f1', 0):.4f}")
    else:
        print(f"\nErro no treinamento: {training_result.get('error', 'Erro desconhecido')}")
        
    return analyzer.is_trained

def demo_sentiment_analysis(analyzer, preprocessor):
    """Demonstra a análise de sentimento em exemplos de textos financeiros."""
    print_section("Demonstração de Análise de Sentimento")
    
    # Exemplos de textos para análise
    exemplos = [
        # Exemplo positivo
        ("""
FATO RELEVANTE: A Empresa XYZ S.A. (XYZS3) comunica aos seus acionistas e ao mercado em geral que
registrou um crescimento significativo de 25% no lucro líquido do segundo trimestre de 2023,
alcançando R$ 450 milhões. Este resultado positivo foi impulsionado pela expansão das operações
internacionais e pelo aumento da eficiência operacional, que elevou a margem EBITDA para 32%.
""", "Comunicado positivo com crescimento expressivo"),
        
        # Exemplo negativo
        ("""
A Empresa ABC S.A. (ABCD3) divulga seus resultados referentes ao primeiro trimestre de 2023.
A receita líquida sofreu uma queda de 12% em relação ao mesmo período do ano anterior,
totalizando R$ 780 milhões. O EBITDA apresentou redução de 20%, refletindo o aumento de custos
operacionais e a deterioração das margens. A companhia registrou prejuízo líquido de R$ 45 milhões,
revertendo o lucro de R$ 30 milhões obtido no 1T22.
""", "Relatório com resultados negativos e prejuízo"),
        
        # Exemplo neutro
        ("""
COMUNICADO AO MERCADO: A Empresa DEF S.A. (DEFS3) informa aos seus acionistas e ao mercado em geral que
mantém suas projeções (guidance) para o ano de 2023, com crescimento esperado entre 3% e 5% na receita
líquida e margem EBITDA estável entre 22% e 24%, em linha com o comunicado ao mercado divulgado em
15 de fevereiro de 2023.
""", "Comunicado neutro mantendo projeções"),
        
        # Exemplo com negações (positivo com elementos negativos)
        ("""
A Empresa GHI S.A. (GHIS3) informa que, apesar do cenário econômico desafiador, não houve queda
nas suas vendas durante o último trimestre. Pelo contrário, a empresa conseguiu manter o crescimento
sustentável e não registrou nenhuma redução em suas margens operacionais, superando as expectativas
dos analistas que previam resultados estáveis ou ligeiramente negativos.
""", "Texto com negações que invertem sentimento"),
        
        # Exemplo com intensificadores
        ("""
Os resultados da Empresa JKL S.A. (JKLS3) foram extremamente decepcionantes neste trimestre,
com vendas muito abaixo do esperado e custos significativamente mais altos. A margem EBITDA caiu
fortemente para apenas 10%, representando uma redução substancial em relação aos 18% do trimestre anterior.
""", "Texto com intensificadores de sentimento negativo")
    ]
    
    for texto, descricao in exemplos:
        print(f"\n{'-' * 40}")
        print(f"EXEMPLO: {descricao}")
        print(f"{'-' * 40}")
        print(texto.strip())
        print(f"{'-' * 40}")
        
        # Preprocessar o texto
        print("\nPreprocessando texto...")
        preprocessed = preprocessor.process(texto)
        
        # Analisar sentimento
        print("Analisando sentimento...")
        result = analyzer.predict(preprocessed)
        
        # Mostrar resultado
        print("\nRESULTADO DA ANÁLISE:")
        print_result(result)
        
        # Gerar explicação detalhada
        print("\nEXPLICAÇÃO DETALHADA:")
        explanation = analyzer.get_sentiment_explanation(preprocessed)
        
        # Mostrar resumo da explicação
        print(explanation["summary"])
        
        # Mostrar fatores contribuintes
        if explanation["contributing_factors"]:
            print("\nFatores que contribuíram para esta classificação:")
            for factor in explanation["contributing_factors"]:
                print(f"- {factor['description']}:")
                if "terms" in factor:
                    print(f"  Termos: {', '.join(factor['terms'])}")
                elif "patterns" in factor:
                    print(f"  Padrões: {', '.join(factor['patterns'])}")
                    
        # Mostrar análise de probabilidade
        if "probability_analysis" in explanation:
            analysis = explanation["probability_analysis"]
            print(f"\nNível de confiança: {analysis['confidence_level']}")
            print(f"Margem entre {analysis['top_class']} ({analysis['top_probability']:.2%}) e")
            print(f"{analysis['second_class']} ({analysis['second_probability']:.2%}): {analysis['margin']:.2%}")
            
        # Perguntar se deseja continuar
        if exemplos.index((texto, descricao)) < len(exemplos) - 1:
            input("\nPressione Enter para o próximo exemplo...")

def benchmark_performance(analyzer, preprocessor, num_repeats=10):
    """Realiza benchmark de performance do analisador."""
    print_section("Benchmark de Performance")
    
    test_text = """
A Empresa XYZ S.A. (XYZS3) divulga seus resultados do primeiro trimestre de 2023. A receita líquida atingiu
R$ 1,2 bilhão, um aumento de 8% em relação ao mesmo período do ano anterior. O EBITDA foi de R$ 350 milhões,
representando uma margem de 29,2%. O lucro líquido alcançou R$ 180 milhões, 5% acima do 1T22.
"""
    
    # Preprocessar o texto uma vez
    preprocessed = preprocessor.process(test_text)
    
    print(f"Realizando {num_repeats} predições para benchmark...")
    
    # Medir tempo de predição
    start_time = time.time()
    for _ in range(num_repeats):
        result = analyzer.predict(preprocessed)
    total_time = time.time() - start_time
    
    avg_time = (total_time / num_repeats) * 1000  # em milissegundos
    
    print(f"Tempo total para {num_repeats} predições: {total_time:.2f} segundos")
    print(f"Tempo médio por predição: {avg_time:.2f} ms")
    print(f"Taxa de processamento: {num_repeats/total_time:.2f} textos/segundo")
    
    # Retornar estatísticas
    return {
        "num_repeats": num_repeats,
        "total_time": total_time,
        "avg_time_ms": avg_time,
        "texts_per_second": num_repeats/total_time
    }

def optimize_parameters(analyzer, preprocessor):
    """Demonstra a otimização de hiperparâmetros."""
    print_section("Otimização de Hiperparâmetros")
    
    # Verificar se o modelo está treinado
    if not analyzer.is_trained:
        print("O modelo precisa estar treinado para otimização de parâmetros.")
        return False
        
    print("Preparando dados para otimização...")
    
    # Dados de treinamento para otimização
    training_texts = [
        "A empresa registrou crescimento de 15% no lucro líquido.",
        "O prejuízo líquido aumentou 20% no trimestre.",
        "A receita ficou estável em relação ao trimestre anterior.",
        "As ações subiram 8% após o anúncio dos resultados.",
        "A empresa teve sua classificação de crédito rebaixada.",
        "O guidance para o próximo ano foi mantido sem alterações.",
        "A margem EBITDA caiu de 25% para 18% no período.",
        "A empresa anunciou aumento nos dividendos.",
        "Os custos operacionais cresceram acima da inflação.",
        "O Conselho aprovou nova política de remuneração sem mudanças significativas."
    ]
    
    training_labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "negative", "positive", "negative", "neutral"
    ]
    
    # Preprocessar textos
    preprocessed_data = []
    for text in training_texts:
        preprocessed = preprocessor.process(text)
        preprocessed_data.append(preprocessed)
        
    # Extrair características
    X = analyzer._prepare_training_features(preprocessed_data)
    y = training_labels
    
    print("Iniciando otimização de hiperparâmetros...")
    
    # Definir grade de parâmetros
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0],
        'norm': [True, False],
        'fit_prior': [True, False]
    }
    
    # Realizar otimização
    results = analyzer.optimize_hyperparameters(X, y, param_grid, cv=3)
    
    print("\nOtimização concluída!")
    print(f"Melhores parâmetros: {results['best_params']}")
    print(f"Melhor pontuação (f1_weighted): {results['best_score']:.4f}")
    
    print("\nParâmetros atualizados no modelo:")
    print(f" - Alpha: {analyzer.alpha}")
    print(f" - Normalização: {analyzer.norm}")
    print(f" - Fit Prior: {analyzer.fit_prior}")
    
    return True

def demo_sentiment_analyzer():
    """Função principal que executa a demonstração completa."""
    print_section("Demonstração do Analisador de Sentimento usando Complement Naive Bayes")
    
    try:
        # Inicializar componentes
        analyzer = create_sentiment_analyzer()
        preprocessor = create_preprocessor()
        
        # Verificar se já existe um modelo treinado
        model_path = os.path.join("models", f"{analyzer.name}.pkl")
        if os.path.exists(model_path):
            print(f"\nModelo existente encontrado em {model_path}.")
            while True:
                choice = input("Deseja carregar o modelo existente? (s/n): ").strip().lower()
                if choice in ['s', 'n']:
                    break
                print("Por favor, digite 's' para sim ou 'n' para não.")
                
            if choice == 's':
                print(f"Carregando modelo existente de {model_path}...")
                success = analyzer.load_model(model_path)
                if success:
                    print("Modelo carregado com sucesso!")
                else:
                    print("Falha ao carregar o modelo. Será necessário treinar um novo modelo.")
                    train_sentiment_analyzer(analyzer, preprocessor)
            else:
                print("Treinando novo modelo...")
                train_sentiment_analyzer(analyzer, preprocessor)
        else:
            print("Nenhum modelo existente encontrado. Treinando novo modelo...")
            train_sentiment_analyzer(analyzer, preprocessor)
            
        # Verificar se o modelo está treinado antes de prosseguir
        if not analyzer.is_trained:
            print("Erro: O modelo não está treinado. A demonstração não pode continuar.")
            return
            
        # Demonstrar análise de sentimento
        demo_sentiment_analysis(analyzer, preprocessor)
        
        # Oferecer opções para benchmark e otimização
        print_section("Funcionalidades Avançadas")
        print("Selecione uma opção:")
        print("1. Benchmark de Performance")
        print("2. Otimização de Hiperparâmetros")
        print("3. Pular e ir para análise interativa")
        
        while True:
            choice = input("Opção (1-3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Opção inválida. Por favor, escolha entre 1 e 3.")
            
        if choice == '1':
            benchmark_performance(analyzer, preprocessor)
        elif choice == '2':
            optimize_parameters(analyzer, preprocessor)
            
        # Demonstração interativa
        print_section("Análise Interativa de Sentimento")
        print("Digite textos financeiros para análise de sentimento (ou 'sair' para encerrar):")
        
        while True:
            print("\nDigite um texto financeiro (ou 'sair'):")
            texto = input("> ").strip()
            
            if texto.lower() == 'sair':
                break
                
            if not texto:
                print("Texto vazio. Por favor, tente novamente.")
                continue
                
            # Preprocessar e analisar
            try:
                preprocessed = preprocessor.process(texto)
                result = analyzer.predict(preprocessed)
                
                # Mostrar resultado
                print("\nResultado da análise:")
                print_result(result)
                
                # Perguntar se deseja ver explicação detalhada
                choice = input("\nDeseja ver explicação detalhada? (s/n): ").strip().lower()
                
                if choice == 's':
                    explanation = analyzer.get_sentiment_explanation(preprocessed)
                    print("\n" + explanation["summary"])
                    
                    if explanation["contributing_factors"]:
                        print("\nFatores contribuintes:")
                        for factor in explanation["contributing_factors"]:
                            print(f"- {factor['description']}")
                            
                    if "probability_analysis" in explanation:
                        analysis = explanation["probability_analysis"]
                        print(f"\nNível de confiança: {analysis['confidence_level']}")
            except Exception as e:
                print(f"Erro ao analisar texto: {str(e)}")
                
        print("\nDemonstração interativa encerrada.")
        
        # Salvar o modelo se treinado ou modificado
        if analyzer.is_trained:
            while True:
                choice = input("\nDeseja salvar o modelo? (s/n): ").strip().lower()
                if choice in ['s', 'n']:
                    break
                print("Por favor, digite 's' para sim ou 'n' para não.")
                
            if choice == 's':
                os.makedirs("models", exist_ok=True)
                model_path = os.path.join("models", f"{analyzer.name}.pkl")
                analyzer.save_model(model_path)
                print(f"Modelo salvo em: {model_path}")
                
    except Exception as e:
        logger.error(f"Erro durante a demonstração: {str(e)}")
        print(f"\nOcorreu um erro: {str(e)}")
        
    print_section("Demonstração Concluída")

if __name__ == "__main__":
    demo_sentiment_analyzer()
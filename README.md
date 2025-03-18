# FinSemble: Sistema de Análise de Texto Financeiro

FinSemble é um framework avançado de análise de texto financeiro baseado em múltiplos classificadores Naive Bayes e técnicas de meta-aprendizado.

## Visão Geral

O sistema utiliza uma arquitetura de ensemble para analisar comunicações financeiras em múltiplas dimensões:
- Classificação de tipo/categoria
- Análise de sentimento
- Modelagem de impacto
- Extração de tópicos e entidades

## Arquitetura

O FinSemble é composto por cinco componentes principais:

1. **Preprocessador Universal** - Prepara os dados textuais para análise
2. **Ensemble de Classificadores Especializados** - Analisa diferentes aspectos do texto
3. **Agregador Bayesiano** - Combina resultados usando meta-aprendizado
4. **Motor de Contextualização** - Refina análises com base em fatores externos
5. **Interface de Explicabilidade** - Fornece justificativas para as previsões

## Instalação

### Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

### Configuração do Ambiente

```bash
# Clonar o repositório
git clone https://github.com/seu-usuario/finsemble.git
cd finsemble

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Instalar o pacote em modo de desenvolvimento
pip install -e .
```

### Configuração Adicional

```bash
# Baixar modelos do spaCy para português
python -m spacy download pt_core_news_lg

# Baixar recursos do NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Uso

*Documentação detalhada de uso será adicionada conforme o desenvolvimento avança.*

## Desenvolvimento

### Estrutura do Projeto
```
finsemble/
├── config/                 # Configurações do sistema
├── data/                   # Datasets e recursos
│   ├── raw/                # Dados brutos
│   ├── processed/          # Dados processados
│   └── knowledge_base/     # Base de conhecimento financeiro
├── src/                    # Código-fonte
│   ├── preprocessor/       # Preprocessador Universal
│   ├── classifiers/        # Classificadores Especializados
│   ├── aggregator/         # Agregador Bayesiano
│   ├── contextualizer/     # Motor de Contextualização
│   ├── explainer/          # Interface de Explicabilidade
│   └── utils/              # Utilitários comuns
├── tests/                  # Testes automatizados
├── notebooks/              # Jupyter notebooks para análise
└── docs/                   # Documentação
```

### Executando Testes

```bash
# Executar todos os testes
pytest

# Executar testes unitários
pytest tests/unit

# Executar testes com cobertura
pytest --cov=src tests/
```

## Licença

*Adicionar informações de licença*

## Contribuição

*Adicionar diretrizes de contribuição*
"""
Módulo base do Preprocessador Universal do FinSemble.

Este módulo contém a implementação do componente responsável pelo
processamento de textos financeiros, incluindo normalização, tokenização
e extração de características para os classificadores especializados.
"""

import re
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple

import nltk
import spacy
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# Importação condicional para validação
try:
    from src.utils.validators import sanitize_text, sanitize_metadata, ValidationError
    VALIDATORS_AVAILABLE = True
except ImportError:
    VALIDATORS_AVAILABLE = False
    
# Importação condicional para gerenciamento de recursos
try:
    from src.utils.resource_manager import ResourceManager
    RESOURCE_MANAGER_AVAILABLE = True
except ImportError:
    RESOURCE_MANAGER_AVAILABLE = False

# Configuração de logging
logger = logging.getLogger(__name__)


class TextProcessor(ABC):
    """Classe abstrata base para processadores de texto."""
    
    @abstractmethod
    def process(self, text: str) -> Any:
        """
        Processa o texto fornecido.
        
        Args:
            text: Texto para processar
            
        Returns:
            Resultado do processamento
        """
        pass


class Normalizer(TextProcessor):
    """
    Normaliza o texto aplicando transformações como lowercase,
    remoção de pontuação, números e stopwords.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o normalizador com as configurações especificadas.
        
        Args:
            config: Dicionário com configurações para normalização
                - lowercase (bool): Converter para minúsculas
                - remove_punctuation (bool): Remover pontuação
                - remove_numbers (bool): Remover números
                - remove_stopwords (bool): Remover stopwords
                - lemmatize (bool): Aplicar lematização
                - language (str): Idioma para stopwords e lematização
        """
        self.config = config
        self.language = config.get("language", "portuguese")
        
        # Preparar recursos necessários
        if config.get("remove_stopwords", True):
            try:
                self.stopwords = set(stopwords.words(self.language))
            except LookupError:
                nltk.download('stopwords')
                self.stopwords = set(stopwords.words(self.language))
                
        if config.get("lemmatize", True):
            try:
                self.lemmatizer = WordNetLemmatizer()
                # Verificar recursos do WordNet
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
                self.lemmatizer = WordNetLemmatizer()
    
    def process(self, text: str) -> str:
        """
        Normaliza o texto aplicando as transformações configuradas.
        
        Args:
            text: Texto para normalizar
            
        Returns:
            Texto normalizado
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Texto inválido para normalização: {text}")
            return ""
        
        # Converter para minúsculas
        if self.config.get("lowercase", True):
            text = text.lower()
            
        # Remover pontuação
        if self.config.get("remove_punctuation", True):
            text = re.sub(r'[^\w\s]', ' ', text)
            
        # Remover números
        if self.config.get("remove_numbers", False):
            text = re.sub(r'\d+', ' ', text)
            
        # Tokenização para processamentos em nível de token
        tokens = word_tokenize(text)
        
        # Remover stopwords
        if self.config.get("remove_stopwords", True):
            tokens = [token for token in tokens if token not in self.stopwords]
            
        # Lematização
        if self.config.get("lemmatize", True):
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
        # Reconstruir texto
        normalized_text = ' '.join(tokens)
        
        # Remover espaços extras
        normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
        
        return normalized_text


class Tokenizer(TextProcessor):
    """
    Realiza a tokenização multi-granular do texto, incluindo
    divisão por documento, seção, parágrafo, sentença e palavras.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o tokenizador com as configurações especificadas.
        
        Args:
            config: Dicionário com configurações para tokenização
                - sentence_split (bool): Dividir em sentenças
                - word_split (bool): Dividir em palavras
                - ngram_range (tuple): Range de n-gramas (min, max)
                - language (str): Idioma para tokenização
        """
        self.config = config
        self.language = config.get("language", "portuguese")
        
        # Preparar recursos necessários
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Carregar modelo do spaCy para análise mais sofisticada
        try:
            if self.language == "portuguese":
                self.nlp = spacy.load("pt_core_news_sm")
            elif self.language == "english":
                self.nlp = spacy.load("en_core_web_sm")
            else:
                logger.warning(f"Idioma {self.language} não suportado pelo spaCy, usando modelo inglês")
                self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(f"Modelo do spaCy não encontrado. Use python -m spacy download pt_core_news_sm")
            self.nlp = None
    
    def process(self, text: str) -> Dict[str, Any]:
        """
        Tokeniza o texto em múltiplos níveis de granularidade.
        
        Args:
            text: Texto para tokenizar
            
        Returns:
            Dicionário com os diferentes níveis de tokenização
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Texto inválido para tokenização: {text}")
            return {"sentences": [], "words": [], "ngrams": []}
        
        result = {}
        
        # Divisão em sentenças
        if self.config.get("sentence_split", True):
            if self.nlp:
                doc = self.nlp(text)
                result["sentences"] = [sent.text for sent in doc.sents]
            else:
                result["sentences"] = sent_tokenize(text, language=self.language)
        
        # Divisão em palavras
        if self.config.get("word_split", True):
            result["words"] = word_tokenize(text, language=self.language)
            
        # Geração de n-gramas
        ngram_range = self.config.get("ngram_range", (1, 3))
        result["ngrams"] = self._generate_ngrams(result.get("words", []), ngram_range)
        
        # Análise sintática adicional caso o spaCy esteja disponível
        if self.nlp:
            doc = self.nlp(text)
            result["entities"] = [(ent.text, ent.label_) for ent in doc.ents]
            result["pos_tags"] = [(token.text, token.pos_) for token in doc]
            
        return result
    
    def _generate_ngrams(self, tokens: List[str], ngram_range: Tuple[int, int]) -> List[str]:
        """
        Gera n-gramas a partir da lista de tokens.
        
        Args:
            tokens: Lista de tokens
            ngram_range: Tuple com intervalo de n-gramas (min, max)
            
        Returns:
            Lista de n-gramas
        """
        min_n, max_n = ngram_range
        ngrams_list = []
        
        for n in range(min_n, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngrams_list.append(' '.join(tokens[i:i + n]))
                
        return ngrams_list


class FeatureExtractor(TextProcessor):
    """
    Extrai características específicas para cada classificador
    a partir do texto processado.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o extrator de características com as configurações especificadas.
        
        Args:
            config: Dicionário com configurações para extração
                - feature_types (list): Tipos de características a extrair
                - max_features (int): Número máximo de características
                - min_df (float): Frequência mínima de documento
                - language (str): Idioma para processamento
        """
        self.config = config
        self.language = config.get("language", "portuguese")
        self.feature_types = config.get("feature_types", ["text"])
        
    def process(self, 
                text: str, 
                normalized_text: Optional[str] = None, 
                tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai características do texto para alimentar os classificadores.
        
        Args:
            text: Texto original
            normalized_text: Texto normalizado (opcional)
            tokens: Resultado da tokenização (opcional)
            
        Returns:
            Dicionário com as características extraídas para cada classificador
        """
        features = {}
        
        # Características para o classificador de tipo (Bernoulli NB)
        if "type_features" in self.feature_types:
            features["type_features"] = self._extract_type_features(
                normalized_text or text, tokens
            )
            
        # Características para o analisador de sentimento (Multinomial NB)
        if "sentiment_features" in self.feature_types:
            features["sentiment_features"] = self._extract_sentiment_features(
                normalized_text or text, tokens
            )
            
        # Características para o modelador de impacto (Gaussian NB)
        if "impact_features" in self.feature_types:
            features["impact_features"] = self._extract_impact_features(
                text, normalized_text, tokens
            )
            
        # Características para o extrator de tópicos e entidades
        if "topic_features" in self.feature_types:
            features["topic_features"] = self._extract_topic_features(
                normalized_text or text, tokens
            )
            
        return features
    
    def _extract_type_features(self, 
                              text: str, 
                              tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai características binárias para classificação de tipo de documento.
        
        Args:
            text: Texto (normalizado de preferência)
            tokens: Estrutura de tokenização (opcional)
            
        Returns:
            Dicionário com características binárias
        """
        features = {}
        
        # Indicadores de tipo de documento financeiro
        type_indicators = {
            "relatorio_trimestral": ["trimestral", "resultado", "trimestre"],
            "comunicado_mercado": ["fato relevante", "comunicado", "mercado"],
            "anuncio_dividendos": ["dividendo", "juros sobre capital", "pagamento"],
            "previsao_resultado": ["guidance", "projeção", "expectativa"],
            "analise_tecnica": ["análise técnica", "suporte", "resistência"],
        }
        
        # Verificar presença dos indicadores
        for doc_type, indicators in type_indicators.items():
            features[f"is_{doc_type}"] = any(indicator in text.lower() for indicator in indicators)
            
        # Características estruturais
        if tokens and "sentences" in tokens:
            features["num_sentences"] = len(tokens["sentences"])
            
        return features
    
    def _extract_sentiment_features(self, 
                                   text: str, 
                                   tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai características para análise de sentimento.
        
        Args:
            text: Texto (normalizado de preferência)
            tokens: Estrutura de tokenização (opcional)
            
        Returns:
            Dicionário com características relacionadas a sentimento
        """
        features = {}
        
        # Contagem simples de termos positivos/negativos
        # (Será substituído por léxicos financeiros mais sofisticados)
        positive_terms = ["aumento", "crescimento", "lucro", "positivo", "alto", 
                         "bom", "melhora", "oportunidade", "superar"]
        negative_terms = ["queda", "redução", "prejuízo", "negativo", "baixo", 
                         "ruim", "piora", "risco", "abaixo"]
        
        if tokens and "words" in tokens:
            words = tokens["words"]
            features["positive_count"] = sum(word in positive_terms for word in words)
            features["negative_count"] = sum(word in negative_terms for word in words)
            features["positive_ratio"] = features["positive_count"] / len(words) if words else 0
            features["negative_ratio"] = features["negative_count"] / len(words) if words else 0
            
        # Análise de intensificadores e negações
        if tokens and "ngrams" in tokens:
            intensifiers = ["muito", "bastante", "extremamente", "significativamente"]
            negations = ["não", "sem", "nunca", "nenhum"]
            
            features["intensifier_count"] = sum(any(i in ng for i in intensifiers) for ng in tokens["ngrams"])
            features["negation_count"] = sum(any(n in ng for n in negations) for ng in tokens["ngrams"])
        
        return features
    
    def _extract_impact_features(self, 
                                original_text: str, 
                                normalized_text: Optional[str] = None,
                                tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai características numéricas para modelagem de impacto.
        
        Args:
            original_text: Texto original
            normalized_text: Texto normalizado (opcional)
            tokens: Estrutura de tokenização (opcional)
            
        Returns:
            Dicionário com características numéricas para impacto
        """
        features = {}
        text = normalized_text or original_text
        
        # Extração de números e percentuais do texto original
        percentage_pattern = r'(\d+(?:,\d+)?%)|(\d+(?:\.\d+)?%)'
        percentage_matches = re.findall(percentage_pattern, original_text)
        features["percentage_count"] = len(percentage_matches)
        
        currency_pattern = r'R\$\s*\d+(?:[,.]\d+)*|\$\s*\d+(?:[,.]\d+)*|€\s*\d+(?:[,.]\d+)*'
        currency_matches = re.findall(currency_pattern, original_text)
        features["currency_count"] = len(currency_matches)
        
        # Características de texto relacionadas a impacto
        impact_indicators = {
            "alta_magnitude": ["significativo", "expressivo", "substancial", "forte"],
            "media_magnitude": ["moderado", "médio", "parcial"],
            "baixa_magnitude": ["leve", "pequeno", "sutil", "ligeiro"]
        }
        
        for impact_type, indicators in impact_indicators.items():
            features[f"impact_{impact_type}"] = sum(text.lower().count(ind) for ind in indicators)
            
        # Características estatísticas do texto
        if tokens and "sentences" in tokens:
            sent_lengths = [len(s.split()) for s in tokens["sentences"]]
            features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0
            features["max_sentence_length"] = max(sent_lengths) if sent_lengths else 0
            
        return features
    
    def _extract_topic_features(self, 
                               text: str, 
                               tokens: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extrai características para identificação de tópicos e entidades.
        
        Args:
            text: Texto (normalizado de preferência)
            tokens: Estrutura de tokenização (opcional)
            
        Returns:
            Dicionário com características relacionadas a tópicos
        """
        features = {}
        
        # Categorias de tópicos financeiros
        topic_categories = {
            "operacional": ["operação", "produção", "capacidade", "eficiência"],
            "financeiro": ["financeiro", "balanço", "patrimônio", "liquidez"],
            "mercado": ["mercado", "concorrência", "setor", "indústria"],
            "governanca": ["governança", "conselho", "administração", "diretoria"],
            "estrategia": ["estratégia", "plano", "expansão", "crescimento"],
            "risco": ["risco", "incerteza", "volatilidade", "exposição"],
            "regulatorio": ["regulação", "regulatório", "lei", "norma", "conformidade"],
            "inovacao": ["inovação", "tecnologia", "pesquisa", "desenvolvimento"],
        }
        
        # Calcular frequências de tópicos
        for topic, keywords in topic_categories.items():
            count = sum(text.lower().count(kw) for kw in keywords)
            features[f"topic_{topic}_count"] = count
            features[f"topic_{topic}_ratio"] = count / len(text.split()) if text else 0
        
        # Extrair entidades se disponíveis na tokenização
        if tokens and "entities" in tokens:
            # Conta entidades por tipo
            entity_types = {}
            for ent_text, ent_type in tokens["entities"]:
                entity_types[ent_type] = entity_types.get(ent_type, 0) + 1
                
            for ent_type, count in entity_types.items():
                features[f"entity_{ent_type}_count"] = count
                
        return features


class PreprocessorUniversal:
    """
    Preprocessador Universal do sistema FinSemble, responsável pelo
    processamento de textos financeiros para os classificadores especializados.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o preprocessador com as configurações especificadas.
        
        Args:
            config: Dicionário com configurações para os componentes
                - normalizer: Configurações do normalizador
                - tokenizer: Configurações do tokenizador
                - feature_extractor: Configurações do extrator de características
                - language: Idioma principal (padrão: "portuguese")
                - fallback_enabled: Habilitar fallbacks para componentes (padrão: True)
                - performance_monitoring: Habilitar monitoramento de performance (padrão: False)
        """
        self.config = config
        self.language = config.get("language", "portuguese")
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.performance_monitoring = config.get("performance_monitoring", False)
        
        # Verificar e inicializar recursos externos, se disponível
        if RESOURCE_MANAGER_AVAILABLE:
            self.resource_manager = ResourceManager(config.get("resources", {}))
            resources = self.resource_manager.initialize_resources()
            
            # Usar o melhor modelo spaCy disponível
            if resources["spacy"]["status"] and resources["spacy"].get("best_model"):
                self.spacy_model = resources["spacy"]["best_model"]
                config.setdefault("tokenizer", {})["spacy_model"] = self.spacy_model
            
            logger.info(f"Recursos inicializados: NLTK={resources['nltk']['status']}, " +
                       f"spaCy={resources['spacy']['status']}")
        
        # Inicializar componentes de processamento
        self.normalizer = Normalizer(config.get("normalizer", {}))
        self.tokenizer = Tokenizer(config.get("tokenizer", {}))
        self.feature_extractor = FeatureExtractor(config.get("feature_extractor", {}))
        
        # Contador para estatísticas de processamento
        self.stats = {
            "texts_processed": 0,
            "errors": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0
        }
        
        logger.info("Preprocessador Universal inicializado com sucesso")
    
    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processa um texto financeiro, aplicando normalização, tokenização
        e extração de características.
        
        Args:
            text: Texto a ser processado
            metadata: Metadados adicionais do documento (opcional)
            
        Returns:
            Dicionário com os resultados do processamento, incluindo:
            - texto original
            - texto normalizado
            - estruturas de tokenização
            - características extraídas para cada classificador
            - metadados originais e derivados
            - estatísticas de processamento, se monitoramento estiver ativado
        """
        start_time = time.time() if self.performance_monitoring else None
        
        # Validar e sanitizar inputs, se disponível
        if VALIDATORS_AVAILABLE:
            try:
                text = sanitize_text(text)
                metadata = sanitize_metadata(metadata)
            except Exception as e:
                logger.warning(f"Erro na validação de inputs: {str(e)}")
                # Continuar com os valores originais em caso de erro
        
        if not text:
            logger.warning(f"Texto vazio ou inválido para processamento")
            return {"error": "Texto inválido ou vazio", "original_text": text, "metadata": metadata or {}}
        
        result = {
            "original_text": text,
            "metadata": metadata or {}
        }
        
        try:
            # Normalização
            try:
                normalized_text = self.normalizer.process(text)
                result["normalized_text"] = normalized_text
            except Exception as e:
                logger.error(f"Erro na normalização: {str(e)}")
                if self.fallback_enabled:
                    # Fallback: usar texto original com normalizações mínimas
                    normalized_text = text.lower() if text else ""
                    result["normalized_text"] = normalized_text
                    result["warnings"] = result.get("warnings", []) + ["Erro na normalização, usando fallback"]
                else:
                    raise
            
            # Tokenização
            try:
                tokens = self.tokenizer.process(normalized_text)
                result["tokens"] = tokens
            except Exception as e:
                logger.error(f"Erro na tokenização: {str(e)}")
                if self.fallback_enabled:
                    # Fallback: tokenização básica
                    basic_tokens = normalized_text.split()
                    tokens = {
                        "words": basic_tokens,
                        "sentences": [normalized_text],
                        "ngrams": basic_tokens
                    }
                    result["tokens"] = tokens
                    result["warnings"] = result.get("warnings", []) + ["Erro na tokenização, usando fallback"]
                else:
                    raise
            
            # Extração de características
            try:
                features = self.feature_extractor.process(
                    text=text,
                    normalized_text=normalized_text,
                    tokens=tokens
                )
                result["features"] = features
            except Exception as e:
                logger.error(f"Erro na extração de características: {str(e)}")
                if self.fallback_enabled:
                    # Fallback: características mínimas
                    features = {
                        "type_features": {"text_length": len(text)},
                        "sentiment_features": {},
                        "impact_features": {},
                        "topic_features": {}
                    }
                    result["features"] = features
                    result["warnings"] = result.get("warnings", []) + ["Erro na extração de características, usando fallback"]
                else:
                    raise
            
            # Gerar metadados derivados
            try:
                result["derived_metadata"] = self._derive_metadata(text, normalized_text, tokens, metadata)
            except Exception as e:
                logger.error(f"Erro ao derivar metadados: {str(e)}")
                if self.fallback_enabled:
                    # Fallback: metadados mínimos
                    result["derived_metadata"] = {"text_length": len(text)}
                    result["warnings"] = result.get("warnings", []) + ["Erro ao derivar metadados, usando fallback"]
                else:
                    raise
            
            # Atualizar estatísticas
            self.stats["texts_processed"] += 1
            
            # Adicionar métricas de performance, se ativado
            if self.performance_monitoring and start_time:
                processing_time = time.time() - start_time
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["texts_processed"]
                )
                
                result["processing_stats"] = {
                    "processing_time_ms": round(processing_time * 1000, 2)
                }
            
            return result
            
        except Exception as e:
            # Registrar erro e atualizar estatísticas
            logger.error(f"Erro ao processar texto: {str(e)}")
            self.stats["errors"] += 1
            
            # Incluir detalhes do erro no resultado
            error_result = {
                "error": str(e), 
                "original_text": text, 
                "metadata": metadata or {}
            }
            
            # Adicionar qualquer resultado parcial que tenha sido gerado
            for key in ["normalized_text", "tokens", "features", "derived_metadata"]:
                if key in result:
                    error_result[key] = result[key]
                    
            # Adicionar métricas de performance, se ativado
            if self.performance_monitoring and start_time:
                error_result["processing_stats"] = {
                    "processing_time_ms": round((time.time() - start_time) * 1000, 2),
                    "status": "error"
                }
                
            return error_result
    
    def process_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Processa um lote de textos financeiros.
        
        Args:
            texts: Lista de textos a serem processados
            metadatas: Lista de metadados para cada texto (opcional)
            
        Returns:
            Lista de resultados de processamento
        """
        if not metadatas:
            metadatas = [None] * len(texts)
        elif len(metadatas) != len(texts):
            logger.warning(f"Número de metadados ({len(metadatas)}) diferente do número de textos ({len(texts)})")
            metadatas = metadatas[:len(texts)] + [None] * (len(texts) - len(metadatas))
            
        results = []
        for text, metadata in zip(texts, metadatas):
            results.append(self.process(text, metadata))
            
        return results
    
    def _derive_metadata(self, 
                        original_text: str, 
                        normalized_text: str, 
                        tokens: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Deriva metadados adicionais a partir do texto e da tokenização.
        
        Args:
            original_text: Texto original
            normalized_text: Texto normalizado
            tokens: Estrutura de tokenização
            metadata: Metadados originais (opcional)
            
        Returns:
            Dicionário com metadados derivados
        """
        derived = {
            "text_length": len(original_text),
            "word_count": len(normalized_text.split()) if normalized_text else 0,
        }
        
        if "sentences" in tokens:
            derived["sentence_count"] = len(tokens["sentences"])
            
        if "entities" in tokens:
            derived["entity_count"] = len(tokens["entities"])
            
        # Detecção básica de complexidade do texto
        if derived["sentence_count"] > 0 and derived["word_count"] > 0:
            # Índice de legibilidade simplificado (aproximação do Flesch para português)
            avg_sentence_length = derived["word_count"] / derived["sentence_count"]
            derived["complexity_score"] = avg_sentence_length
            
            # Classificação de complexidade
            if avg_sentence_length > 25:
                derived["complexity_level"] = "alta"
            elif avg_sentence_length > 15:
                derived["complexity_level"] = "média"
            else:
                derived["complexity_level"] = "baixa"
                
        return derived
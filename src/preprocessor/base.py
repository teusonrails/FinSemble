"""
Módulo para preprocessamento modular de textos financeiros.

Implementa uma arquitetura de pipeline flexível para transformação
e enriquecimento de textos financeiros.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, Type
from collections import OrderedDict

# Configuração de logging
logger = logging.getLogger(__name__)

class PreprocessingStep(ABC):
    """Classe base abstrata para etapas do pipeline de preprocessamento."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa a etapa de preprocessamento.
        
        Args:
            config: Configurações específicas para esta etapa
        """
        self.config = config or {}
        self.name = self.config.get("name", self.__class__.__name__)
        self.enabled = self.config.get("enabled", True)
    
    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa os dados de entrada e retorna os dados transformados.
        
        Args:
            data: Dicionário com dados para processamento
            
        Returns:
            Dicionário com dados processados
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} (enabled={self.enabled})"


class PreprocessingPipeline:
    """
    Pipeline configurável para processar textos financeiros através
    de uma sequência de etapas de preprocessamento.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o pipeline de preprocessamento.
        
        Args:
            config: Configurações para o pipeline
        """
        self.config = config or {}
        self.name = self.config.get("name", "DefaultPipeline")
        self.steps = OrderedDict()
        self.fallback_enabled = self.config.get("fallback_enabled", True)
        self.stop_on_error = self.config.get("stop_on_error", False)
        self.performance_monitoring = self.config.get("performance_monitoring", False)
        
        # Estatísticas
        self.stats = {
            "processed_count": 0,
            "error_count": 0,
            "warning_count": 0
        }
        
        logger.info(f"Pipeline de preprocessamento '{self.name}' inicializado")
    
    def add_step(self, step_id: str, step: PreprocessingStep) -> 'PreprocessingPipeline':
        """
        Adiciona uma etapa ao pipeline.
        
        Args:
            step_id: Identificador único da etapa
            step: Instância da etapa de preprocessamento
            
        Returns:
            Self para encadeamento de métodos
        """
        if step_id in self.steps:
            logger.warning(f"Substituindo etapa existente com ID '{step_id}'")
            
        self.steps[step_id] = step
        logger.debug(f"Etapa '{step_id}' ({step.__class__.__name__}) adicionada ao pipeline '{self.name}'")
        return self
    
    def remove_step(self, step_id: str) -> 'PreprocessingPipeline':
        """
        Remove uma etapa do pipeline.
        
        Args:
            step_id: Identificador da etapa a remover
            
        Returns:
            Self para encadeamento de métodos
        """
        if step_id in self.steps:
            del self.steps[step_id]
            logger.debug(f"Etapa '{step_id}' removida do pipeline '{self.name}'")
        else:
            logger.warning(f"Tentativa de remover etapa inexistente '{step_id}'")
            
        return self
    
    def get_step(self, step_id: str) -> Optional[PreprocessingStep]:
        """
        Obtém uma etapa do pipeline por ID.
        
        Args:
            step_id: Identificador da etapa
            
        Returns:
            A etapa, se existir, ou None
        """
        return self.steps.get(step_id)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa os dados através de todas as etapas do pipeline.
        
        Args:
            data: Dicionário com dados para processamento
            
        Returns:
            Dicionário com dados processados
        """
        import time
        
        if not data:
            logger.warning("Dados vazios fornecidos para pipeline de preprocessamento")
            return {"error": "Dados vazios", "error_code": "EMPTY_INPUT"}
            
        self.stats["processed_count"] += 1
        
        # Inicializar resultado com dados originais
        result = data.copy()
        result["pipeline"] = self.name
        result["warnings"] = []
        
        # Contagem de tempo, se monitoramento de performance ativado
        step_times = {}
        start_time = time.time() if self.performance_monitoring else None
        
        # Processar cada etapa
        for step_id, step in self.steps.items():
            if not step.enabled:
                logger.debug(f"Pulando etapa desativada '{step_id}'")
                continue
                
            step_start = time.time() if self.performance_monitoring else None
            
            try:
                logger.debug(f"Executando etapa '{step_id}'")
                step_result = step.process(result)
                
                # Atualizar resultado com o resultado da etapa
                result.update(step_result)
                
            except Exception as e:
                logger.error(f"Erro na etapa '{step_id}': {str(e)}", exc_info=True)
                result["warnings"].append(f"Erro na etapa '{step_id}': {str(e)}")
                self.stats["error_count"] += 1
                
                if self.fallback_enabled:
                    logger.info(f"Usando fallback para etapa '{step_id}'")
                else:
                    if self.stop_on_error:
                        result["error"] = f"Erro na etapa '{step_id}': {str(e)}"
                        result["error_code"] = f"STEP_ERROR_{step_id.upper()}"
                        return result
                        
            if self.performance_monitoring and step_start:
                step_time = time.time() - step_start
                step_times[step_id] = step_time
                logger.debug(f"Etapa '{step_id}' executada em {step_time:.4f} segundos")
        
        # Remover lista de avisos se vazia
        if not result["warnings"]:
            del result["warnings"]
        else:
            self.stats["warning_count"] += len(result["warnings"])
        
        # Adicionar estatísticas de performance, se ativado
        if self.performance_monitoring and start_time:
            result["processing_stats"] = {
                "total_time": time.time() - start_time,
                "step_times": step_times
            }
            
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do pipeline.
        
        Returns:
            Dicionário com estatísticas de processamento
        """
        return self.stats.copy()


class TextNormalizer(PreprocessingStep):
    """Implementação modular do normalizador de texto."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Verificar se texto está presente
        if "original_text" not in data:
            logger.warning("Texto original não encontrado para normalização")
            return {"normalized_text": ""}
            
        text = data["original_text"]
        if not text:
            return {"normalized_text": ""}
            
        # Aplicar normalização
        lowercase = self.config.get("lowercase", True)
        remove_punctuation = self.config.get("remove_punctuation", True)
        remove_numbers = self.config.get("remove_numbers", False)
        
        # Aplicar transformações
        if lowercase:
            text = text.lower()
            
        if remove_punctuation:
            import re
            text = re.sub(r'[^\w\s]', ' ', text)
            
        if remove_numbers:
            import re
            text = re.sub(r'\d+', ' ', text)
            
        # Remover espaços extras
        text = re.sub(r'\s+', ' ', text).strip()
        
        return {
            "normalized_text": text
        }


class FinancialTextTokenizer(PreprocessingStep):
    """Tokenizador especializado para textos financeiros."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Verificar se há texto normalizado
        text = data.get("normalized_text", data.get("original_text", ""))
        if not text:
            return {"tokens": {"sentences": [], "words": []}}
            
        # Aplicar tokenização
        import nltk
        
        # Garantir recursos do NLTK disponíveis
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        # Tokenização em sentenças e palavras
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Gerar n-gramas
        ngram_range = self.config.get("ngram_range", (1, 3))
        min_n, max_n = ngram_range
        
        ngrams = []
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(' '.join(words[i:i + n]))
                
        return {
            "tokens": {
                "sentences": sentences,
                "words": words,
                "ngrams": ngrams
            }
        }


class EntityExtractor(PreprocessingStep):
    """Extrator de entidades para textos financeiros."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Verificar se há texto para processar
        text = data.get("normalized_text", data.get("original_text", ""))
        if not text:
            return {"entities": []}
            
        # Carregar modelo spaCy se configurado
        import spacy
        
        # Usar modelo compatível com o idioma
        language = self.config.get("language", "portuguese")
        
        try:
            if language == "portuguese":
                nlp = spacy.load("pt_core_news_sm")
            elif language == "english":
                nlp = spacy.load("en_core_web_sm")
            else:
                logger.warning(f"Idioma {language} não suportado pelo spaCy, usando modelo inglês")
                nlp = spacy.load("en_core_web_sm")
                
            # Processar texto com spaCy
            doc = nlp(text)
            
            # Extrair entidades
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            pos_tags = [(token.text, token.pos_) for token in doc]
            
            return {
                "entities": entities,
                "pos_tags": pos_tags
            }
            
        except Exception as e:
            logger.warning(f"Erro ao extrair entidades: {str(e)}")
            return {"entities": [], "pos_tags": []}


class FinancialFeatureExtractor(PreprocessingStep):
    """Extrator de características específicas para textos financeiros."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Verificar disponibilidade de texto e tokens
        text = data.get("normalized_text", data.get("original_text", ""))
        tokens = data.get("tokens", {})
        
        if not text:
            return {"features": {}}
            
        # Extrair diferentes tipos de características
        features = {}
        
        # Características para classificação de tipo
        type_features = self._extract_type_features(text, tokens)
        
        # Características para análise de sentimento
        sentiment_features = self._extract_sentiment_features(text, tokens)
        
        # Características para modelagem de impacto
        impact_features = self._extract_impact_features(text, tokens)
        
        # Consolidar todas as características
        features["type_features"] = type_features
        features["sentiment_features"] = sentiment_features
        features["impact_features"] = impact_features
        
        return {"features": features}
    
    def _extract_type_features(self, text: str, tokens: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        # Verificar presença de padrões indicativos de tipo de documento
        type_indicators = {
            "relatorio": ["relatório", "trimestral", "anual", "resultados"],
            "comunicado": ["comunicado", "fato relevante", "mercado"],
            "guidance": ["projeção", "guidance", "previsão", "expectativa"]
        }
        
        for doc_type, indicators in type_indicators.items():
            features[f"is_{doc_type}"] = any(indicator in text.lower() for indicator in indicators)
            
        return features
    
    def _extract_sentiment_features(self, text: str, tokens: Dict[str, Any]) -> Dict[str, Any]:
        features = {}
        
        # Análise básica de termos positivos/negativos
        positive_terms = ["aumento", "crescimento", "lucro", "positivo"]
        negative_terms = ["queda", "redução", "prejuízo", "negativo"]
        
        features["positive_count"] = sum(term in text.lower() for term in positive_terms)
        features["negative_count"] = sum(term in text.lower() for term in negative_terms)
        
        return features
    
    def _extract_impact_features(self, text: str, tokens: Dict[str, Any]) -> Dict[str, Any]:
        import re
        features = {}
        
        # Extrair percentuais
        percentage_pattern = r'(\d+(?:,\d+)?%)|(\d+(?:\.\d+)?%)'
        percentage_matches = re.findall(percentage_pattern, text)
        features["percentage_count"] = len(percentage_matches)
        
        # Extrair valores monetários
        currency_pattern = r'R\$\s*\d+(?:[,.]\d+)*|\$\s*\d+(?:[,.]\d+)*|€\s*\d+(?:[,.]\d+)*'
        currency_matches = re.findall(currency_pattern, text)
        features["currency_count"] = len(currency_matches)
        
        return features


class MetadataEnricher(PreprocessingStep):
    """Enriquece os metadados com informações derivadas."""
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Verificar disponibilidade de texto
        text = data.get("normalized_text", data.get("original_text", ""))
        tokens = data.get("tokens", {})
        metadata = data.get("metadata", {})
        
        derived_metadata = {}
        
        # Características básicas do texto
        derived_metadata["text_length"] = len(text)
        derived_metadata["word_count"] = len(tokens.get("words", []))
        derived_metadata["sentence_count"] = len(tokens.get("sentences", []))
        
        # Métricas de complexidade
        if derived_metadata["sentence_count"] > 0 and derived_metadata["word_count"] > 0:
            avg_sentence_length = derived_metadata["word_count"] / derived_metadata["sentence_count"]
            derived_metadata["complexity_score"] = avg_sentence_length
            
            # Classificação de complexidade
            if avg_sentence_length > 25:
                derived_metadata["complexity_level"] = "alta"
            elif avg_sentence_length > 15:
                derived_metadata["complexity_level"] = "média"
            else:
                derived_metadata["complexity_level"] = "baixa"
                
        return {"derived_metadata": derived_metadata}


def create_default_pipeline(config: Dict[str, Any] = None) -> PreprocessingPipeline:
    """
    Cria um pipeline padrão com etapas comuns para textos financeiros.
    
    Args:
        config: Configurações para o pipeline
        
    Returns:
        Pipeline de preprocessamento configurado
    """
    config = config or {}
    pipeline_config = config.get("pipeline", {})
    
    # Criar pipeline
    pipeline = PreprocessingPipeline(pipeline_config)
    
    # Adicionar etapas padrão
    pipeline.add_step("normalizer", TextNormalizer(config.get("normalizer", {})))
    pipeline.add_step("tokenizer", FinancialTextTokenizer(config.get("tokenizer", {})))
    pipeline.add_step("entity_extractor", EntityExtractor(config.get("entity_extractor", {})))
    pipeline.add_step("feature_extractor", FinancialFeatureExtractor(config.get("feature_extractor", {})))
    pipeline.add_step("metadata_enricher", MetadataEnricher(config.get("metadata_enricher", {})))
    
    return pipeline

class PreprocessorUniversal:
    """
    Versão refatorada do preprocessador universal do sistema FinSemble.
    
    Esta implementação utiliza o novo sistema de pipeline modular para
    processamento mais flexível e robusto de textos financeiros.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o preprocessador com a configuração especificada.
        
        Args:
            config: Configurações para o preprocessador
        """
        self.config = config
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.performance_monitoring = config.get("performance_monitoring", False)
        
        # Inicializar o sistema de cache
        cache_config = config.get("cache", {})
        self.cache = PreprocessingCache(cache_config)
        
        # Flag para processamento seletivo
        self.selective_processing = config.get("selective_processing", True)
        
        # Inicializar pipeline principal
        self.pipeline = create_default_pipeline(config)
        
        # Inicializar pipelines especializados por tipo de classificador
        self.specialized_pipelines = {
            "type": self._create_type_pipeline(config),
            "sentiment": self._create_sentiment_pipeline(config),
            "impact": self._create_impact_pipeline(config)
        }
        
        # Estatísticas
        self.stats = {
            "texts_processed": 0,
            "errors": 0,
            "total_processing_time": 0,
            "avg_processing_time": 0
        }
        
        logger.info("PreprocessorUniversal refatorado inicializado com sucesso")
    
    def _create_type_pipeline(self, config: Dict[str, Any]) -> PreprocessingPipeline:
        """Cria pipeline otimizado para classificação de tipo."""
        pipeline_config = config.copy()
        pipeline_config["name"] = "TypePipeline"
        
        pipeline = create_default_pipeline(pipeline_config)
        
        # Otimizar para classificação de tipo (por exemplo, ignorar análise de sentimento)
        feature_extractor = pipeline.get_step("feature_extractor")
        if feature_extractor:
            feature_extractor.config["extract_sentiment"] = False
            feature_extractor.config["extract_impact"] = False
            
        return pipeline
    
    def _create_sentiment_pipeline(self, config: Dict[str, Any]) -> PreprocessingPipeline:
        """Cria pipeline otimizado para análise de sentimento."""
        pipeline_config = config.copy()
        pipeline_config["name"] = "SentimentPipeline"
        
        pipeline = create_default_pipeline(pipeline_config)
        
        # Otimizar para análise de sentimento
        normalizer = pipeline.get_step("normalizer")
        if normalizer:
            normalizer.config["remove_numbers"] = True  # Números geralmente não são relevantes para sentimento
            
        return pipeline
    
    def _create_impact_pipeline(self, config: Dict[str, Any]) -> PreprocessingPipeline:
        """Cria pipeline otimizado para modelagem de impacto."""
        pipeline_config = config.copy()
        pipeline_config["name"] = "ImpactPipeline"
        
        pipeline = create_default_pipeline(pipeline_config)
        
        # Otimizar para modelagem de impacto
        normalizer = pipeline.get_step("normalizer")
        if normalizer:
            normalizer.config["remove_numbers"] = False  # Números são cruciais para impacto
            normalizer.config["remove_punctuation"] = False  # Símbolos monetários e percentuais são importantes
            
        return pipeline
    
    def process(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                pipeline_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa um texto financeiro, aplicando normalização, tokenização
        e extração de características, com otimizações de cache.
        
        Args:
            text: Texto a ser processado
            metadata: Metadados adicionais do documento (opcional)
            processing_requirements: Requisitos específicos de processamento (opcional)
                - normalize: Se deve normalizar o texto
                - tokenize: Se deve tokenizar o texto
                - extract_features: Se deve extrair características
                - extract_type_features: Se deve extrair características de tipo
                - extract_sentiment_features: Se deve extrair características de sentimento
                - extract_impact_features: Se deve extrair características de impacto
            
        Returns:
            Dicionário com os resultados do processamento
        """
        
        start_time = time.time() if self.performance_monitoring else None
        
        # Validar e sanitizar inputs
        if VALIDATORS_AVAILABLE:
            try:
                text = sanitize_text(text)
                metadata = sanitize_metadata(metadata)
            except Exception as e:
                logger.warning(f"Erro na validação de inputs: {str(e)}")
        
        if not text:
            logger.warning(f"Texto vazio ou inválido para processamento")
            return {"error": "Texto inválido ou vazio", "original_text": text, "metadata": metadata or {}}
        
        # Configurar requisitos de processamento
        req = processing_requirements or {}
        normalize_text = req.get("normalize", True)
        tokenize_text = req.get("tokenize", True)
        extract_features = req.get("extract_features", True)
        
        # Configurações para extração seletiva de características
        extract_type = req.get("extract_type_features", extract_features)
        extract_sentiment = req.get("extract_sentiment_features", extract_features)
        extract_impact = req.get("extract_impact_features", extract_features)
        
        result = {
            "original_text": text,
            "metadata": metadata or {}
        }
        
        try:
            # Normalização com cache
            if normalize_text:
                normalized_text = None
                
                # Tentar obter do cache
                cached_normalized = self.cache.get_normalized_text(
                    text,
                    self.config.get("normalizer", {})
                )
                
                if cached_normalized:
                    normalized_text = cached_normalized
                    logger.debug("Usando texto normalizado do cache")
                else:
                    # Se não estiver em cache, normalizar e armazenar
                    try:
                        normalized_text = self.normalizer.process(text)
                        
                        # Armazenar no cache
                        self.cache.cache_normalized_text(
                            text,
                            normalized_text,
                            self.config.get("normalizer", {})
                        )
                    except Exception as e:
                        logger.error(f"Erro na normalização: {str(e)}")
                        if self.fallback_enabled:
                            normalized_text = text.lower() if text else ""
                            result["warnings"] = result.get("warnings", []) + ["Erro na normalização, usando fallback"]
                        else:
                            raise
                
                result["normalized_text"] = normalized_text
            
            # Tokenização com cache
            if tokenize_text:
                tokens = None
                text_for_tokens = result.get("normalized_text", text)
                
                # Tentar obter do cache
                cached_tokens = self.cache.get_tokens(
                    text_for_tokens,
                    self.config.get("tokenizer", {})
                )
                
                if cached_tokens:
                    tokens = cached_tokens
                    logger.debug("Usando tokens do cache")
                else:
                    # Se não estiver em cache, tokenizar e armazenar
                    try:
                        tokens = self.tokenizer.process(text_for_tokens)
                        
                        # Armazenar no cache
                        self.cache.cache_tokens(
                            text_for_tokens,
                            tokens,
                            self.config.get("tokenizer", {})
                        )
                    except Exception as e:
                        logger.error(f"Erro na tokenização: {str(e)}")
                        if self.fallback_enabled:
                            basic_tokens = text_for_tokens.split()
                            tokens = {
                                "words": basic_tokens,
                                "sentences": [text_for_tokens],
                                "ngrams": basic_tokens
                            }
                            result["warnings"] = result.get("warnings", []) + ["Erro na tokenização, usando fallback"]
                        else:
                            raise
                
                result["tokens"] = tokens
            
            # Extração de características com cache e seletividade
            if extract_features:
                features = {}
                
                # Extração seletiva de características por tipo
                feature_configs = {
                    "type": self.config.get("feature_extractor", {}).get("type_features", {}),
                    "sentiment": self.config.get("feature_extractor", {}).get("sentiment_features", {}),
                    "impact": self.config.get("feature_extractor", {}).get("impact_features", {})
                }
                
                # Texto para extração de características
                text_for_features = result.get("normalized_text", text)
                
                # Extrair características de tipo
                if extract_type:
                    cached_type_features = self.cache.get_features(
                        text_for_features,
                        "type",
                        feature_configs["type"]
                    )
                    
                    if cached_type_features:
                        features["type_features"] = cached_type_features
                        logger.debug("Usando características de tipo do cache")
                    else:
                        try:
                            type_features = self.feature_extractor._extract_type_features(
                                text_for_features,
                                result.get("tokens", {})
                            )
                            features["type_features"] = type_features
                            
                            self.cache.cache_features(
                                text_for_features,
                                "type",
                                type_features,
                                feature_configs["type"]
                            )
                        except Exception as e:
                            logger.error(f"Erro na extração de características de tipo: {str(e)}")
                            if self.fallback_enabled:
                                features["type_features"] = {"text_length": len(text)}
                                result["warnings"] = result.get("warnings", []) + ["Erro na extração de características de tipo, usando fallback"]
                            else:
                                raise
                
                # Extrair características de sentimento
                if extract_sentiment:
                    cached_sentiment_features = self.cache.get_features(
                        text_for_features,
                        "sentiment",
                        feature_configs["sentiment"]
                    )
                    
                    if cached_sentiment_features:
                        features["sentiment_features"] = cached_sentiment_features
                        logger.debug("Usando características de sentimento do cache")
                    else:
                        try:
                            sentiment_features = self.feature_extractor._extract_sentiment_features(
                                text_for_features,
                                result.get("tokens", {})
                            )
                            features["sentiment_features"] = sentiment_features
                            
                            self.cache.cache_features(
                                text_for_features,
                                "sentiment",
                                sentiment_features,
                                feature_configs["sentiment"]
                            )
                        except Exception as e:
                            logger.error(f"Erro na extração de características de sentimento: {str(e)}")
                            if self.fallback_enabled:
                                features["sentiment_features"] = {}
                                result["warnings"] = result.get("warnings", []) + ["Erro na extração de características de sentimento, usando fallback"]
                            else:
                                raise
                
                # Extrair características de impacto
                if extract_impact:
                    cached_impact_features = self.cache.get_features(
                        text_for_features,
                        "impact",
                        feature_configs["impact"]
                    )
                    
                    if cached_impact_features:
                        features["impact_features"] = cached_impact_features
                        logger.debug("Usando características de impacto do cache")
                    else:
                        try:
                            impact_features = self.feature_extractor._extract_impact_features(
                                text_for_features,
                                result.get("tokens", {})
                            )
                            features["impact_features"] = impact_features
                            
                            self.cache.cache_features(
                                text_for_features,
                                "impact",
                                impact_features,
                                feature_configs["impact"]
                            )
                        except Exception as e:
                            logger.error(f"Erro na extração de características de impacto: {str(e)}")
                            if self.fallback_enabled:
                                features["impact_features"] = {}
                                result["warnings"] = result.get("warnings", []) + ["Erro na extração de características de impacto, usando fallback"]
                            else:
                                raise
                
                result["features"] = features
            
            # Gerar metadados derivados
            try:
                result["derived_metadata"] = self._derive_metadata(
                    text, 
                    result.get("normalized_text", ""), 
                    result.get("tokens", {}), 
                    metadata
                )
            except Exception as e:
                logger.error(f"Erro ao derivar metadados: {str(e)}")
                if self.fallback_enabled:
                    result["derived_metadata"] = {"text_length": len(text)}
                    result["warnings"] = result.get("warnings", []) + ["Erro ao derivar metadados, usando fallback"]
                else:
                    raise
            
            # Atualizar estatísticas
            self.stats["texts_processed"] += 1
            
            # Métricas de performance
            if self.performance_monitoring and start_time:
                processing_time = time.time() - start_time
                self.stats["total_processing_time"] += processing_time
                self.stats["avg_processing_time"] = (
                    self.stats["total_processing_time"] / self.stats["texts_processed"]
                )
                
                result["processing_stats"] = {
                    "processing_time_ms": round(processing_time * 1000, 2),
                    "cache_stats": self.cache.get_stats()
                }
            
            return result
            
        except Exception as e:
            # Registrar erro e atualizar estatísticas
            logger.error(f"Erro não tratado ao processar texto: {str(e)}", exc_info=True)
            self.stats["errors"] += 1
            
            # Retornar informações de erro
            error_result = {
                "error": str(e),
                "error_code": "PROCESSING_ERROR",
                "original_text": text,
                "metadata": metadata or {}
            }
            
            # Adicionar tempo de processamento, se monitoramento ativado
            if self.performance_monitoring and start_time:
                error_result["processing_stats"] = {
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "status": "error"
                }
                
            return error_result
    
    def process_batch(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None,
                    pipeline_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Processa um lote de textos financeiros.
        
        Args:
            texts: Lista de textos a serem processados
            metadatas: Lista de metadados para cada texto (opcional)
            pipeline_type: Tipo de pipeline a usar (None = pipeline padrão)
            
        Returns:
            Lista de resultados de processamento
        """
        # Normalizar metadados
        if metadatas is None:
            metadatas = [None] * len(texts)
        elif len(metadatas) != len(texts):
            logger.warning(f"Número de metadados ({len(metadatas)}) diferente do número de textos ({len(texts)})")
            if len(metadatas) < len(texts):
                metadatas = metadatas + [None] * (len(texts) - len(metadatas))
            else:
                metadatas = metadatas[:len(texts)]
                
        # Otimização para processamento em lote
        # Implementar processamento paralelo para lotes grandes
        import concurrent.futures
        
        # Usar ThreadPoolExecutor para paralelização
        # (adequado para operações com I/O como NLP)
        max_workers = min(len(texts), os.cpu_count() * 2)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Criar tarefas para o executor
            futures = [
                executor.submit(self.process, text, metadata, requirements)
                for text, metadata in zip(texts, metadatas)
            ]
            
            # Coletar resultados na mesma ordem
            results = [future.result() for future in futures]
        
        return results
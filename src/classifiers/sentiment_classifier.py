"""
Módulo para análise de sentimento de textos financeiros.

Este módulo implementa o Analisador de Sentimento do FinSemble,
utilizando o algoritmo Complement Naive Bayes para identificar
o sentimento (positivo, negativo, neutro) em comunicações financeiras.
"""

import os
import re
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import GridSearchCV

from src.utils.feature_manager import FeatureManager
from src.classifiers.base import BaseClassifier
from src.utils.feature_extractors import (
    LexiconFeatureExtractor, 
    StructuralFeatureExtractor,
    TemporalFeatureExtractor,
    CompositeFeatureExtractor
)
from src.utils.lexicon_manager import LexiconManager
from src.utils.performance_utils import timed, cached_feature_extraction, batch_process, optimize_array_operations

# Configuração de logging
logger = logging.getLogger(__name__)

class WeightNormalizedComplementNB(ComplementNB):
    """
    Implementação personalizada do Complement Naive Bayes com normalização de peso.
    
    Esta classe estende o ComplementNB padrão adicionando transformação logarítmica
    e normalização dos vetores de peso para melhorar o desempenho em dados desbalanceados.
    """
    
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, norm=True):
        """
        Inicializa o classificador Complement NB com normalização de peso.
        
        Args:
            alpha: Parâmetro de suavização Laplaciana (smoothing)
            fit_prior: Se deve aprender probabilidades de classe de dados
            class_prior: Probabilidades a priori para as classes
            norm: Se deve aplicar normalização l2 aos vetores de peso
        """
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior, norm=norm)
        
    @optimize_array_operations
    def _count_alpha(self, X, y):
        """
        Calcula pesos de características para cada classe usando contagens complementares.
        
        Esta função sobrescreve o método padrão para adicionar transformação logarítmica
        e normalização aos pesos.
        
        Args:
            X: Matriz de características
            y: Rótulos de classe
            
        Returns:
            Matriz de pesos por classe
        """
        # Obter classes únicas e suas contagens
        self.classes_, y_inv = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        
        # Calcular contagens de classe
        n_samples, n_features = X.shape
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        np.add.at(self.class_count_, y_inv, 1)
        
        # Soma das características para cada classe
        feature_count = np.zeros((n_classes, n_features), dtype=np.float64)
        np.add.at(feature_count, y_inv, X)
        
        # Soma total de características
        total_feature_count = feature_count.sum(axis=0)
        
        # Complement counts (contagens de todas as classes exceto a atual)
        comp_feature_count = total_feature_count - feature_count
        comp_class_count = n_samples - self.class_count_
        
        # Aplicar suavização de Laplace
        feature_count += self.alpha
        comp_feature_count += self.alpha * n_classes
        
        # Calcular probabilidades de características complementares
        with np.errstate(divide='ignore', invalid='ignore'):
            feature_prob = comp_feature_count / comp_class_count[:, np.newaxis]
        
        # Aplicar transformação logarítmica para atenuar diferenças extremas
        # log(1+x) é mais estável que log(x) pois evita log(0) = -inf
        feature_prob = np.log(1.0 + feature_prob)
        
        # Normalizar os vetores de peso para cada classe
        if self.norm:
            feature_prob = normalize(feature_prob, axis=1, norm='l2')
            
        return feature_prob

class SentimentAnalyzer(BaseClassifier):
    """
    Analisador de sentimento para textos financeiros.
    
    Utiliza o algoritmo Complement Naive Bayes para classificar documentos com base
    em características textuais e léxicas para determinar sentimento.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o analisador de sentimento.
        
        Args:
            config: Configurações específicas para o classificador, incluindo:
                - alpha: Parâmetro de suavização para Complement NB
                - fit_prior: Se deve aprender probabilidades de classe
                - norm: Se deve aplicar normalização aos vetores de peso
                - feature_extraction: Configurações para extração de características
                - class_mapping: Mapeamento entre códigos de classe e nomes legíveis
                - sentiment_lexicon: Léxico de sentimento financeiro personalizado
                - lexicon_path: Caminho para arquivo de léxico externo
        """
        # Chamar inicializador da classe base
        super().__init__(config)
        
        # Configurações específicas para Complement NB
        self.alpha = config.get("alpha", 1.0)
        self.fit_prior = config.get("fit_prior", True)
        self.norm = config.get("norm", True)
        
        # Configurações para extração de características
        self.feature_extraction = config.get("feature_extraction", {})
        self.max_features = self.feature_extraction.get("max_features", 5000)
        self.min_df = self.feature_extraction.get("min_df", 3)
        self.ngram_range = tuple(self.feature_extraction.get("ngram_range", (1, 3)))
        self.use_idf = self.feature_extraction.get("use_idf", True)
        
        # Inicializar vetorizador e transformador TF-IDF
        self.vectorizer = None
        self.tfidf_transformer = None
        
        # Mapeamento de classes para nomes legíveis
        self.class_mapping = config.get("class_mapping", {
            "positive": "Positivo",
            "negative": "Negativo",
            "neutral": "Neutro"
        })
        
        # Inicializar gerenciador de léxicos
        lexicon_config = {
            "lexicon_path": config.get("lexicon_path"),
        }
        self.lexicon_manager = LexiconManager(lexicon_config)
        
        # Carregar léxico de sentimento
        self.sentiment_lexicon = config.get("sentiment_lexicon", self.lexicon_manager.get_lexicon())
        
        # Inicializar extratores de características
        self._init_feature_extractors()
        
        # Configurações adicionais
        self.version = "1.1.0"  # Versão semântica para acompanhamento de evolução
        self.metadata = {
            "description": "Analisador de Sentimento com Complement Naive Bayes",
            "parameters": {
                "alpha": self.alpha,
                "fit_prior": self.fit_prior,
                "norm": self.norm,
                "max_features": self.max_features,
                "min_df": self.min_df,
                "ngram_range": self.ngram_range,
                "use_idf": self.use_idf
            },
            "lexicon_stats": {
                "positive_terms": len(self.sentiment_lexicon.get("positive", [])),
                "negative_terms": len(self.sentiment_lexicon.get("negative", [])),
                "neutral_terms": len(self.sentiment_lexicon.get("neutral", []))
            }
        }
        
        logger.info(f"Analisador de Sentimento v{self.version} inicializado com parâmetros: alpha={self.alpha}, "
                    f"fit_prior={self.fit_prior}, norm={self.norm}, max_features={self.max_features}")
        
        # Inicializar o gerenciador de características
        feature_manager_config = config.get("feature_manager", {})
        self.feature_manager = FeatureManager(feature_manager_config)
        
        # Após inicializar o vectorizer, registrá-lo no gerenciador
        if self.vectorizer:
            self.feature_manager.register_vectorizer("sentiment", self.vectorizer)
        
        # Registrar extratores personalizados
        self.feature_manager.register_extractor("sentiment_custom", self._extract_custom_features)
                    
    def _init_feature_extractors(self):
        """Inicializa os extratores de características."""
        
        # Configurações para os extratores
        lexicon_config = {
            "sentiment_lexicon": self.sentiment_lexicon,
            "negation_terms": [
                "não", "nem", "nenhum", "nenhuma", "nunca", "jamais", "tampouco",
                "sem", "exceto", "salvo"
            ],
            "intensifiers": [
                "muito", "extremamente", "significativamente", "altamente", "bastante",
                "consideravelmente", "expressivamente", "fortemente", "substancialmente"
            ],
            "diminishers": [
                "pouco", "ligeiramente", "levemente", "um pouco", "parcialmente",
                "relativamente", "moderadamente", "minimamente"
            ]
        }
        
        temporal_config = {
            "future_indicators": [
                "irá", "deverá", "prevê", "projeta", "espera", "antecipa", "planeja"
            ],
            "past_indicators": [
                "foi", "registrou", "apresentou", "obteve", "alcançou", "realizou"
            ]
        }
        
        # Inicializar extratores individuais
        self.lexicon_extractor = LexiconFeatureExtractor(lexicon_config)
        self.structural_extractor = StructuralFeatureExtractor()
        self.temporal_extractor = TemporalFeatureExtractor(temporal_config)
        
        # Criar extrator composto
        self.feature_extractor = CompositeFeatureExtractor([
            self.lexicon_extractor,
            self.structural_extractor,
            self.temporal_extractor
        ])
        
    def _create_model(self) -> BaseEstimator:
        """
        Cria e configura o modelo Complement Naive Bayes.
        
        Returns:
            Instância configurada do modelo WeightNormalizedComplementNB
        """
        logger.info(f"Criando modelo Complement NB com parâmetros: alpha={self.alpha}, "
                   f"fit_prior={self.fit_prior}, norm={self.norm}")
                   
        model = WeightNormalizedComplementNB(
            alpha=self.alpha,
            fit_prior=self.fit_prior,
            norm=self.norm,
            class_prior=None
        )
        
        return model
        
    def _validate_input(self, preprocessed_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Valida os dados de entrada para processamento.
        
        Args:
            preprocessed_data: Dados preprocessados para validação
            
        Returns:
            Tupla com (é_válido, mensagem_erro)
        """
        if not preprocessed_data:
            error_msg = "Dados preprocessados vazios ou nulos"
            logger.error(error_msg)
            return False, {"error": error_msg, "error_code": "EMPTY_INPUT"}
            
        # Verificar se contém texto necessário
        if "normalized_text" not in preprocessed_data and "original_text" not in preprocessed_data:
            error_msg = "Dados preprocessados não contêm texto (normalized_text ou original_text)"
            logger.error(error_msg)
            return False, {"error": error_msg, "error_code": "MISSING_TEXT"}
            
        return True, None
        
    def _get_text(self, preprocessed_data: Dict[str, Any]) -> str:
        """
        Obtém o texto dos dados preprocessados.
        
        Args:
            preprocessed_data: Dados preprocessados
            
        Returns:
            Texto para processamento
        """
        if "normalized_text" in preprocessed_data:
            return preprocessed_data["normalized_text"]
        elif "original_text" in preprocessed_data:
            return preprocessed_data["original_text"]
        return ""
        
    @timed
    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Extrai características baseadas no texto usando vetorização e TF-IDF.
        
        Args:
            text: Texto para vetorização
            
        Returns:
            Array numpy com características extraídas do texto
        """
        if not self.vectorizer or not hasattr(self.vectorizer, 'vocabulary_'):
            # Em modo de treinamento, retornamos o texto para processamento posterior
            logger.debug("Vectorizer não inicializado, retornando texto para processamento posterior")
            return text
            
        try:
            # Em modo de predição, aplicar vetorização ao texto
            logger.debug(f"Aplicando vetorização ao texto (comprimento: {len(text)})")
            
            # Vetorizar o texto
            counts = self.vectorizer.transform([text])
            
            # Aplicar transformação TF-IDF se configurada
            if self.tfidf_transformer and self.use_idf:
                features = self.tfidf_transformer.transform(counts).toarray()[0]
            else:
                features = counts.toarray()[0]
                
            if logger.isEnabledFor(logging.DEBUG):
                # Log da densidade das características (proporção de valores não-zero)
                non_zero = np.count_nonzero(features)
                logger.debug(f"Características extraídas: {len(features)} total, {non_zero} não-zero "
                            f"({non_zero/len(features)*100:.1f}% de densidade)")
                            
            return features
            
        except Exception as e:
            logger.error(f"Erro na extração de características de texto: {str(e)}")
            
            # Retornar vetor de zeros como fallback
            if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
                return np.zeros((1, len(self.vectorizer.get_feature_names_out())))
            return np.array([])
            
    @cached_feature_extraction(maxsize=256)
    def _extract_custom_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai características customizadas para análise de sentimento.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Dicionário com características extraídas
        """
        # Obter o texto e tokens
        text = self._get_text(preprocessed_data)
        tokens = preprocessed_data.get("tokens", {})
        
        # Usar o extrator composto para extrair todas as características
        return self.feature_extractor.extract(text, tokens=tokens)
        
    @timed
    def _combine_features(self, text_features: np.ndarray, custom_features: Dict[str, Any]) -> np.ndarray:
        """
        Combina características de texto e características customizadas.
        
        Args:
            text_features: Características extraídas do texto
            custom_features: Características customizadas
            
        Returns:
            Array combinado de características
        """
        if not custom_features:
            logger.debug("Sem características customizadas para combinar")
            return text_features
            
        try:
            # Converter para DataFrame para manipulação mais fácil
            custom_df = pd.DataFrame([custom_features])
            
            # Selecionar apenas colunas numéricas e booleanas
            numeric_cols = custom_df.select_dtypes(include=['number', 'bool']).columns
            
            if not numeric_cols.empty:
                X_custom = custom_df[numeric_cols].values[0]
                
                # Verificar compatibilidade de tipo
                if not isinstance(X_custom, np.ndarray):
                    X_custom = np.array(X_custom)
                    
                # Combinar as características
                X_combined = np.concatenate([text_features, X_custom])
                
                logger.debug(f"Características combinadas: {len(text_features)} de texto + "
                           f"{len(X_custom)} customizadas = {len(X_combined)} total")
                           
                return X_combined
            else:
                logger.warning("Nenhuma característica customizada numérica ou booleana encontrada")
                return text_features
                
        except Exception as e:
            logger.error(f"Erro ao combinar características: {str(e)}")
            return text_features
            
    def _extract_features(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extrai características para análise de sentimento dos dados preprocessados.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Array numpy com características para classificação
        """
        # Validar entrada
        is_valid, error = self._validate_input(preprocessed_data)
        if not is_valid:
            logger.error(f"Validação de entrada falhou: {error['error']}")
            return np.array([])
            
        # Obter texto para processamento
        if "normalized_text" in preprocessed_data:
            text = preprocessed_data["normalized_text"]
        else:
            text = preprocessed_data["original_text"]
            
        try:
            # Usar o gerenciador para extração eficiente
            text_features = self.feature_manager.extract_text_features(
                text, "sentiment", as_sparse=False
            )
            
            # Extrair características customizadas através do gerenciador
            custom_features = self.feature_manager.extract_custom_features(
                preprocessed_data, "sentiment_custom"
            )
            
            # Combinar características
            combined_features = self.feature_manager.combine_features(
                text_features, custom_features, force_dense=True
            )
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {str(e)}", exc_info=True)
            return np.array([])
            
    @timed
    def _prepare_training_features(self, training_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Sobrescreve o método da classe base para incluir extração paralela eficiente.
        
        Args:
            training_data: Lista de dados preprocessados
            
        Returns:
            Array numpy com as características para treinamento
        """
        start_time = time.time()
        logger.info(f"Preparando características para treinamento com {len(training_data)} amostras")
        
        # Extrair textos normalizados
        texts = []
        for data in training_data:
            if "normalized_text" in data:
                text = data["normalized_text"]
            elif "original_text" in data:
                text = data["original_text"]
            else:
                logger.warning("Amostra sem texto, usando string vazia")
                text = ""
            texts.append(text)
        
        # Inicializar e treinar o vetorizador com todos os textos
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range
        )
        
        # Vetorizar textos mantendo representação esparsa
        logger.info("Vetorizando textos...")
        X_text = self.vectorizer.fit_transform(texts)
        
        # Registrar vetorizador no gerenciador de características
        self.feature_manager.register_vectorizer("sentiment", self.vectorizer)
        
        # Aplicar transformação TF-IDF se configurada, mantendo representação esparsa
        if self.use_idf:
            logger.info("Aplicando transformação TF-IDF...")
            self.tfidf_transformer = TfidfTransformer()
            X_text = self.tfidf_transformer.fit_transform(X_text)
        
        # Armazenar nomes das características
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Log de informações
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vetorização concluída: {vocab_size} termos extraídos do vocabulário")
        
        # Extração de características customizadas em paralelo
        custom_features_list = []
        
        # Função para extrair características customizadas de uma amostra
        def extract_custom(data):
            return self.feature_manager.extract_custom_features(data, "sentiment_custom")
        
        # Processar em paralelo com ThreadPoolExecutor (mais eficiente para esta tarefa I/O bound)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            custom_features_list = list(executor.map(extract_custom, training_data))
        
        # Combinar todas as características
        # Primeiro, converter características customizadas para formato compatível
        if custom_features_list and any(custom_features_list):
            custom_df = pd.DataFrame(custom_features_list)
            numeric_cols = custom_df.select_dtypes(include=['number', 'bool']).columns
            
            if not numeric_cols.empty:
                X_custom = custom_df[numeric_cols].values
                
                # Combinar características mantendo formato esparso até o final
                if sp.issparse(X_text):
                    X_custom_sparse = sp.csr_matrix(X_custom)
                    X_combined = sp.hstack([X_text, X_custom_sparse])
                else:
                    X_combined = np.hstack((X_text, X_custom))
                
                # Atualizar nomes das características
                self.feature_names = np.concatenate([self.feature_names, np.array(numeric_cols)])
                
                logger.info(f"Adicionadas {len(numeric_cols)} características customizadas")
                logger.info(f"Dimensões finais do conjunto de características: {X_combined.shape}")
                
                # Converter para array denso apenas no final, se necessário
                # Importante: alguns algoritmos trabalham diretamente com matrizes esparsas
                # Verificar se o algoritmo suporta entrada esparsa antes de converter
                if hasattr(self.model, "_validate_data") and not sp.issparse(X_combined):
                    try:
                        # Tentar manter esparso, conversão apenas se necessário
                        return X_combined
                    except:
                        # Se o modelo exigir formato denso, converter
                        return X_combined.toarray() if sp.issparse(X_combined) else X_combined
                else:
                    # Converter para formato denso para compatibilidade geral
                    return X_combined.toarray() if sp.issparse(X_combined) else X_combined
        
        # Se não há características customizadas, apenas retornar X_text
        # Converter para formato denso apenas se necessário
        if hasattr(self.model, "_validate_data") and not sp.issparse(X_text):
            return X_text
        else:
            return X_text.toarray() if sp.issparse(X_text) else X_text
        
    @timed
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sobrescreve o método da classe base para incluir extração de
        características personalizadas na predição de sentimento.
        
        Args:
            data: Dados preprocessados para classificação
            
        Returns:
            Dicionário com a classe prevista e probabilidades
        """
        # Validar estado do modelo
        if not self.is_trained:
            logger.warning(f"Classificador {self.name} não treinado")
            return {"error": "Modelo não treinado", "error_code": "MODEL_NOT_TRAINED"}
            
        # Validar entrada
        is_valid, error = self._validate_input(data)
        if not is_valid:
            return error
            
        # Verificar se o modelo possui as propriedades necessárias
        if not hasattr(self, 'feature_names_') or not hasattr(self, 'classes_'):
            return {"error": "Modelo não está completamente inicializado", "error_code": "MODEL_INITIALIZATION_ERROR"}
        
        start_time = time.time()
        logger.debug(f"Iniciando predição para amostra")
        
        try:
                # Verificar se precisamos fazer preprocessamento adicional
            if "normalized_text" not in data or "tokens" not in data or "features" not in data:
                # Precisamos garantir que tenhamos o preprocessamento necessário
                if "original_text" in data:
                    # Obter preprocessador universal
                    from src.preprocessor.base import preprocessor_instance
                    
                    # Criar requisitos específicos para análise de sentimento
                    requirements = {
                        "normalize": True,
                        "tokenize": True,
                        "extract_features": True,
                        "extract_type_features": False,  # Não precisamos para sentimento
                        "extract_sentiment_features": True,
                        "extract_impact_features": False  # Não precisamos para sentimento
                    }
                    
                    # Fazer preprocessamento seletivo
                    preprocessed = preprocessor_instance.process(
                        data["original_text"],
                        data.get("metadata"),
                        requirements
                    )
                    
                    # Atualizar dados com resultados do preprocessamento
                    data.update(preprocessed)
            # Extrair características
            features = self._extract_features(data)
            
            # Verificar se features não é None
            if features is None:
                return {"error": "Falha na extração de características", "error_code": "FEATURE_EXTRACTION_ERROR"}
                
            # Verificar dimensão das características
            if len(features.shape) < 2:
                # Reshapear para batch de uma amostra
                features = features.reshape(1, -1)
                
            # Verificar compatibilidade de dimensões
            expected_size = len(self.feature_names_)
            actual_size = features.shape[1]
            
            if actual_size != expected_size:
                logger.warning(f"Dimensão incompatível de características: obtido {actual_size}, "
                            f"esperado {expected_size}")
                            
                # Estratégia adaptativa para lidar com diferenças de dimensão
                if actual_size < expected_size:
                    # Preencher com zeros
                    padded = np.zeros((1, expected_size))
                    padded[0, :actual_size] = features[0, :actual_size]
                    features = padded
                else:
                    # Truncar para tamanho esperado
                    features = features[0, :expected_size].reshape(1, -1)
                    
            # Realizar predição
            predicted_class = self.model.predict(features)[0]
            
            # Obter probabilidades (calibradas, se disponíveis)
            if self.calibrated_model and self.calibrate_probabilities:
                probabilities = self.calibrated_model.predict_proba(features)[0]
            else:
                probabilities = self.model.predict_proba(features)[0]
                
            # Criar mapeamento de classes para probabilidades
            class_probs = {str(cls): float(prob) for cls, prob in zip(self.classes_, probabilities)}
            
            # Converter código da classe para nome legível, se disponível
            predicted_class_str = str(predicted_class)
            readable_class = self.class_mapping.get(predicted_class_str, predicted_class_str)
            
            # Atualizar estatísticas
            inference_time = time.time() - start_time
            self.stats["inference_count"] += 1
            self.stats["total_inference_time"] += inference_time
            self.stats["avg_inference_time"] = (
                self.stats["total_inference_time"] / self.stats["inference_count"]
            )
            
            logger.debug(f"Predição concluída em {inference_time*1000:.2f}ms. "
                    f"Classe: {predicted_class_str}, Confiança: {max(probabilities):.4f}")
                    
            # Encontrar termos relevantes para a classificação
            text = self._get_text(data)
            top_terms = self._get_top_terms_for_class(predicted_class_str, text, top_n=5)
            
            # Resultado da predição
            result = {
                "predicted_class": predicted_class_str,
                "readable_class": readable_class,
                "probabilities": class_probs,
                "confidence": float(max(probabilities)),
                "inference_time": inference_time,
                "relevant_terms": top_terms,
                "class_mapping": self.class_mapping  # Adicionar para referência
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}", exc_info=True)
            return {
                "error": f"Erro na predição: {str(e)}",
                "error_code": "PREDICTION_ERROR",
                "traceback": str(e.__traceback__)
            }  
                   
    def _get_top_terms_for_class(self, class_label: str, text: str, top_n: int = 5) -> List[str]:
        """
        Identifica os termos mais relevantes para a classificação de sentimento.
        
        Args:
            class_label: Rótulo da classe prevista
            text: Texto original ou normalizado
            top_n: Número de termos a retornar
            
        Returns:
            Lista de termos relevantes
        """
        if not self.is_trained or not hasattr(self.model, 'feature_log_prob_'):
            logger.warning("Impossível obter termos relevantes: modelo não treinado ou não configurado corretamente")
            return []
            
        try:
            # Encontrar índice da classe
            class_idx = np.where(self.classes_ == class_label)[0]
            if len(class_idx) == 0:
                logger.warning(f"Classe '{class_label}' não encontrada no modelo")
                return []
                
            class_idx = class_idx[0]
            
            # Obter log-probabilidades dos recursos para esta classe
            feature_probs = self.model.feature_log_prob_[class_idx]
            
            # Vetorizar o texto
            text_vector = self.vectorizer.transform([text]).toarray()[0]
            
            # Identificar termos presentes no texto
            present_term_indices = np.where(text_vector > 0)[0]
            
            if len(present_term_indices) == 0:
                logger.debug("Nenhum termo do vocabulário encontrado no texto")
                return []
                
            # Obter pontuações para termos presentes
            term_scores = feature_probs[present_term_indices]
            
            # Classificar por relevância
            sorted_indices = np.argsort(term_scores)[::-1]
            
            # Selecionar os top_n termos mais relevantes
            top_indices = present_term_indices[sorted_indices[:top_n]]
            top_terms = [self.feature_names[i] for i in top_indices]
            
            logger.debug(f"Termos mais relevantes para classe '{class_label}': {top_terms}")
            return top_terms
            
        except Exception as e:
            logger.warning(f"Erro ao identificar termos relevantes: {str(e)}")
            return []
            
    def get_sentiment_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera uma explicação detalhada da análise de sentimento.
        
        Args:
            data: Dados preprocessados ou resultado da predição
            
        Returns:
            Dicionário com explicação detalhada
        """
        # Verificar se já temos uma predição ou precisamos fazer uma
        if "predicted_class" in data and "probabilities" in data:
            prediction = data
        else:
            prediction = self.predict(data)
            
        if "error" in prediction:
            return {"error": prediction["error"]}
            
        # Obter texto
        text = ""
        if isinstance(data, dict):
            if "normalized_text" in data:
                text = data["normalized_text"]
            elif "original_text" in data:
                text = data["original_text"]
                
        if not text and "original_text" in prediction:
            text = prediction["original_text"]
            
        if not text:
            return {"error": "Texto não encontrado para explicação"}
            
        # Preparar explicação
        explanation = {
            "sentiment": prediction["readable_class"],
            "confidence": prediction["confidence"],
            "summary": self._generate_sentiment_summary(prediction, text),
            "contributing_factors": []
        }
        
        # Extrair fatores que contribuem para o sentimento
        try:
            # Termos relevantes do léxico
            sentiment_class = prediction["predicted_class"]
            if sentiment_class in self.sentiment_lexicon:
                text_lower = text.lower()
                sentiment_terms = []
                for term in self.sentiment_lexicon[sentiment_class]:
                    if term.lower() in text_lower:
                        sentiment_terms.append(term)
                        
                if sentiment_terms:
                    explanation["contributing_factors"].append({
                        "type": "sentiment_terms",
                        "description": f"Termos de sentimento {sentiment_class}",
                        "terms": sentiment_terms[:10]  # Limitar a 10 termos
                    })
                    
            # Verificar negações que podem inverter o sentimento
            negation_patterns = []
            for negation in self.lexicon_extractor.negation_terms:
                pattern = fr'{negation}\s+(\w+\s+){{0,5}}(\w+)'
                matches = re.findall(pattern, text.lower())
                if matches:
                    negation_patterns.extend([f"{negation} ... {m[-1]}" for m in matches])
                    
            if negation_patterns:
                explanation["contributing_factors"].append({
                    "type": "negation_patterns",
                    "description": "Padrões de negação que podem modificar o sentimento",
                    "patterns": negation_patterns[:5]  # Limitar a 5 padrões
                })
                
            # Verificar intensificadores
            intensifier_patterns = []
            for intensifier in self.lexicon_extractor.intensifiers:
                pattern = fr'{intensifier}\s+(\w+)'
                matches = re.findall(pattern, text.lower())
                if matches:
                    intensifier_patterns.extend([f"{intensifier} {m}" for m in matches])
                    
            if intensifier_patterns:
                explanation["contributing_factors"].append({
                    "type": "intensifiers",
                    "description": "Intensificadores que amplificam o sentimento",
                    "patterns": intensifier_patterns[:5]  # Limitar a 5 padrões
                })
                
            # Análise comparativa de probabilidades
            probs = prediction["probabilities"]
            if len(probs) > 1:
                sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_probs) >= 2:
                    top_class, top_prob = sorted_probs[0]
                    second_class, second_prob = sorted_probs[1]
                    margin = top_prob - second_prob
                    explanation["probability_analysis"] = {
                        "top_class": self.class_mapping.get(top_class, top_class),
                        "top_probability": top_prob,
                        "second_class": self.class_mapping.get(second_class, second_class),
                        "second_probability": second_prob,
                        "margin": margin,
                        "confidence_level": self._interpret_confidence_margin(margin)
                    }
                    
        except Exception as e:
            logger.error(f"Erro ao gerar explicação detalhada: {str(e)}")
            explanation["error_in_explanation"] = str(e)
            
        return explanation
        
    def _generate_sentiment_summary(self, prediction: Dict[str, Any], text: str) -> str:
        """
        Gera um resumo em linguagem natural do sentimento detectado.
        
        Args:
            prediction: Resultado da predição
            text: Texto analisado
            
        Returns:
            Resumo em texto do sentimento
        """
        sentiment = prediction["readable_class"]
        confidence = prediction["confidence"]
        
        # Classificar o nível de confiança
        confidence_level = "alta" if confidence > 0.8 else "moderada" if confidence > 0.6 else "baixa"
        
        # Gerar texto base com base no sentimento
        if sentiment == "Positivo":
            summary = f"O texto apresenta um sentimento predominantemente positivo (confiança {confidence_level}: {confidence:.1%})"
        elif sentiment == "Negativo":
            summary = f"O texto apresenta um sentimento predominantemente negativo (confiança {confidence_level}: {confidence:.1%})"
        else:  # Neutro
            summary = f"O texto apresenta um sentimento neutro ou balanceado (confiança {confidence_level}: {confidence:.1%})"
            
        # Adicionar termos relevantes se disponíveis
        if "relevant_terms" in prediction and prediction["relevant_terms"]:
            terms = ", ".join(prediction["relevant_terms"])
            summary += f". Termos relevantes: {terms}"
            
        return summary
        
    def _interpret_confidence_margin(self, margin: float) -> str:
        """
        Interpreta a margem de confiança entre as duas principais classes.
        
        Args:
            margin: Diferença entre as probabilidades das duas principais classes
            
        Returns:
            Interpretação em texto da margem
        """
        if margin > 0.5:
            return "muito alta (classificação clara)"
        elif margin > 0.3:
            return "alta (classificação confiável)"
        elif margin > 0.15:
            return "moderada (classificação provável)"
        elif margin > 0.05:
            return "baixa (classificação incerta)"
        else:
            return "muito baixa (classificação ambígua)"
            
    def optimize_hyperparameters(self, X, y, param_grid=None, cv=5, scoring='f1_weighted'):
        """
        Otimiza hiperparâmetros usando validação cruzada.
        
        Args:
            X: Características de treinamento
            y: Rótulos de treinamento
            param_grid: Grade de parâmetros a testar (None para grade padrão)
            cv: Número de folds para validação cruzada
            scoring: Métrica para otimização
            
        Returns:
            Dicionário com melhores parâmetros e pontuação
        """
        if param_grid is None:
            param_grid = {
                'alpha': [0.1, 0.5, 1.0, 2.0],
                'norm': [True, False],
                'fit_prior': [True, False]
            }
            
        logger.info(f"Iniciando otimização de hiperparâmetros com {cv} folds de validação cruzada")
        
        # Criar modelo base
        base_model = WeightNormalizedComplementNB()
        
        # Configurar busca em grade
        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv,
            scoring=scoring, n_jobs=-1,
            verbose=1
        )
        
        # Executar busca
        grid_search.fit(X, y)
        
        # Registrar resultados
        logger.info(f"Otimização concluída. Melhores parâmetros: {grid_search.best_params_}")
        logger.info(f"Melhor pontuação ({scoring}): {grid_search.best_score_:.4f}")
        
        # Atualizar parâmetros do modelo
        self.alpha = grid_search.best_params_['alpha']
        self.norm = grid_search.best_params_['norm']
        self.fit_prior = grid_search.best_params_['fit_prior']
        
        # Retornar resultados
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Salva o modelo treinado em disco com metadados aprimorados.
        
        Args:
            filepath: Caminho para salvar o modelo (opcional)
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        if not self.is_trained:
            logger.warning(f"Tentativa de salvar classificador {self.name} não treinado")
            raise ValueError("Modelo não treinado")
            
        if filepath is None:
            os.makedirs(self.model_path, exist_ok=True)
            filepath = os.path.join(self.model_path, f"{self.name}.pkl")
            
        # Preparar objeto para serialização com metadados aprimorados
        model_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "feature_names": self.feature_names,
            "classes_": self.classes_,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata,
            "stats": self.stats,
            "config": self.config,
            "version": self.version,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_extraction_config": {
                "vectorizer_config": {
                    "max_features": self.max_features,
                    "min_df": self.min_df,
                    "ngram_range": self.ngram_range
                },
                "use_idf": self.use_idf,
                "lexicon_summary": {
                    "positive_terms": len(self.sentiment_lexicon.get("positive", [])),
                    "negative_terms": len(self.sentiment_lexicon.get("negative", [])),
                    "neutral_terms": len(self.sentiment_lexicon.get("neutral", []))
                }
            }
        }
        
        # Salvar arquivo
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Modelo salvo em {filepath} (versão {self.version})")
        return filepath
        
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Carrega um modelo treinado do disco.
        
        Args:
            filepath: Caminho para carregar o modelo (opcional)
            
        Returns:
            True se o carregamento for bem-sucedido, False caso contrário
        """
        if filepath is None:
            filepath = os.path.join(self.model_path, f"{self.name}.pkl")
            
        try:
            import pickle
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # Restaurar estado do modelo
            self.model = model_data["model"]
            self.calibrated_model = model_data.get("calibrated_model")
            self.feature_names = model_data.get("feature_names")
            self.classes_ = model_data.get("classes_")
            self.is_trained = model_data.get("is_trained", True)
            self.training_metadata = model_data.get("training_metadata", {})
            self.stats = model_data.get("stats", self.stats)
            
            # Restaurar configuração, preservando valores originais não presentes no arquivo
            if "config" in model_data:
                for key, value in model_data["config"].items():
                    self.config[key] = value
                    
            # Verificar informações de versão para compatibilidade
            model_version = model_data.get("version", "0.0.0")
            if model_version != self.version:
                logger.warning(f"Versão do modelo ({model_version}) diferente da versão atual ({self.version})")
                
            # Registrar informações de carregamento
            logger.info(f"Modelo carregado de {filepath} (versão {model_version})")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de {filepath}: {str(e)}")
            return False
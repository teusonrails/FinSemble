"""
Módulo para classificação de tipo/categoria de textos financeiros.

Este módulo implementa o Classificador de Tipo/Categoria do FinSemble,
utilizando o algoritmo Bernoulli Naive Bayes para identificar diferentes
tipos de comunicações financeiras (relatórios, comunicados, análises, etc.).
"""

import re
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from collections import Counter

from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator

from src.classifiers.base import BaseClassifier
from src.utils.validators import ValidationError

# Configuração de logging
logger = logging.getLogger(__name__)


class TypeClassifier(BaseClassifier):
    """
    Classificador para identificação do tipo/categoria de comunicações financeiras.
    
    Utiliza o algoritmo Bernoulli Naive Bayes para classificar documentos com base
    em características binárias (presença/ausência de termos-chave).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o classificador de tipo.
        
        Args:
            config: Configurações específicas para o classificador, incluindo:
                - alpha: Parâmetro de suavização (smoothing) para Bernoulli NB
                - binarize: Limite para binarização de características
                - fit_prior: Se deve aprender probabilidades de classe
                - feature_extraction: Configurações para extração de características
                - class_mapping: Mapeamento entre códigos de classe e nomes legíveis
        """
        # Chamar inicializador da classe base
        super().__init__(config)
        
        # Configurações específicas para Bernoulli NB
        self.alpha = config.get("alpha", 1.0)
        self.binarize = config.get("binarize", 0.0)
        self.fit_prior = config.get("fit_prior", True)
        
        # Configurações para extração de características
        self.feature_extraction = config.get("feature_extraction", {})
        self.max_features = self.feature_extraction.get("max_features", 1000)
        self.min_df = self.feature_extraction.get("min_df", 2)
        self.ngram_range = tuple(self.feature_extraction.get("ngram_range", (1, 2)))
        
        # Inicializar vectorizer para extrair características de texto
        self.vectorizer = None
        
        # Mapeamento de classes para nomes legíveis
        self.class_mapping = config.get("class_mapping", {})
        
        # Indicadores-chave para tipos específicos de documentos
        self.type_indicators = {
            "relatorio_trimestral": [
                "relatório trimestral", "resultado trimestral", "demonstrações financeiras", 
                "balanço patrimonial", "trimestre encerrado"
            ],
            "comunicado_mercado": [
                "comunicado ao mercado", "fato relevante", "aviso aos acionistas", 
                "comunicação", "informa ao mercado"
            ],
            "anuncio_dividendos": [
                "distribuição de dividendos", "juros sobre capital", "pagamento de dividendos", 
                "proventos", "jcp", "dividendo"
            ],
            "previsao_resultado": [
                "guidance", "projeção", "estimativa", "perspectiva", 
                "previsão", "outlook", "expectativa"
            ],
            "analise_tecnica": [
                "análise técnica", "suporte", "resistência", "tendência", 
                "indicadores técnicos", "gráfico"
            ],
            "analise_fundamentalista": [
                "análise fundamentalista", "valuation", "múltiplos", "fundamentos", 
                "valor intrínseco", "fluxo de caixa descontado"
            ],
            "ipo_oferta": [
                "oferta pública", "ipo", "emissão de ações", "bookbuilding", 
                "prospecto", "follow-on"
            ],
            "aquisicao_fusao": [
                "aquisição", "fusão", "incorporação", "combinação de negócios", 
                "compra de participação", "m&a"
            ],
            "reestruturacao": [
                "reestruturação", "reorganização societária", "cisão", "spin-off", 
                "recuperação judicial", "reestruturação de dívida"
            ],
            "mudanca_gestao": [
                "novo presidente", "novo ceo", "novo diretor", "mudança na diretoria", 
                "conselho de administração", "eleição de membros"
            ]
        }
        
        logger.info(f"Classificador de Tipo/Categoria inicializado com parâmetros: alpha={self.alpha}, "
                   f"binarize={self.binarize}, fit_prior={self.fit_prior}, max_features={self.max_features}")
        logger.info(f"Configurado para classificar {len(self.type_indicators)} categorias")
    
    def _create_model(self) -> BaseEstimator:
        """
        Cria e configura o modelo Bernoulli Naive Bayes.
        
        Returns:
            Instância configurada do modelo BernoulliNB
        """
        logger.info(f"Criando modelo Bernoulli NB com parâmetros: alpha={self.alpha}, "
                   f"binarize={self.binarize}, fit_prior={self.fit_prior}")
        model = BernoulliNB(
            alpha=self.alpha,
            binarize=self.binarize,
            fit_prior=self.fit_prior,
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
    
    def _extract_text_features(self, text: str) -> np.ndarray:
        """
        Extrai características baseadas no texto usando vetorização.
        
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
            features = self.vectorizer.transform([text]).toarray()[0]
            
            if logger.isEnabledFor(logging.DEBUG):
                # Log da densidade das características (proporção de valores não-zero)
                non_zero = np.count_nonzero(features)
                logger.debug(f"Características extraídas: {len(features)} total, {non_zero} não-zero "
                           f"({non_zero/len(features)*100:.1f}% de densidade)")
            
            return features
        except Exception as e:
            logger.error(f"Erro na extração de características de texto: {str(e)}")
            # Retornar vetor de zeros como fallback
            return np.zeros((1, len(self.vectorizer.get_feature_names_out())))
    
    def _extract_custom_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai características manuais/heurísticas para classificação de tipo.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Dicionário com características extraídas
        """
        features = {}
        
        # Obter o texto normalizado ou original
        if "normalized_text" in preprocessed_data:
            text = preprocessed_data["normalized_text"].lower()
        elif "original_text" in preprocessed_data:
            text = preprocessed_data["original_text"].lower()
        else:
            logger.warning("Dados preprocessados não contêm texto para extração de características customizadas")
            return {}
            
        # Verificar presença dos indicadores de tipo
        indicator_features = self._extract_type_indicator_features(text)
        features.update(indicator_features)
            
        # Características estruturais
        structural_features = self._extract_structural_features(preprocessed_data)
        features.update(structural_features)
                
        # Características contextuais dos metadados
        metadata_features = self._extract_metadata_features(preprocessed_data)
        features.update(metadata_features)
        
        logger.debug(f"Extraídas {len(features)} características customizadas")
        return features
    
    def _extract_type_indicator_features(self, text: str) -> Dict[str, Any]:
        """
        Extrai características baseadas em indicadores de tipo de documento.
        
        Args:
            text: Texto normalizado em minúsculas
            
        Returns:
            Dicionário com características relacionadas a indicadores de tipo
        """
        features = {}
        
        for doc_type, indicators in self.type_indicators.items():
            # Verificar presença de indicadores
            is_present = any(indicator.lower() in text for indicator in indicators)
            features[f"is_{doc_type}"] = is_present
            
            # Contar ocorrências
            count = sum(text.count(indicator.lower()) for indicator in indicators)
            features[f"count_{doc_type}"] = count
            
            # Se presente, verificar posição (indicadores no início têm mais relevância)
            if is_present:
                first_positions = []
                for indicator in indicators:
                    pos = text.find(indicator.lower())
                    if pos >= 0:
                        first_positions.append(pos)
                
                if first_positions:
                    # Normalizar posição pelo comprimento do texto
                    min_pos = min(first_positions)
                    features[f"pos_{doc_type}"] = min_pos / len(text) if len(text) > 0 else 0
        
        return features
    
    def _extract_structural_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai características estruturais do texto.
        
        Args:
            preprocessed_data: Dados preprocessados
            
        Returns:
            Dicionário com características estruturais
        """
        features = {}
        tokens = preprocessed_data.get("tokens", {})
        
        if "sentences" in tokens:
            features["num_sentences"] = len(tokens["sentences"])
            
            # Calcular comprimento médio e variância das sentenças
            sent_lengths = [len(s.split()) for s in tokens["sentences"]]
            features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0
            features["var_sentence_length"] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
            
        if "words" in tokens:
            features["num_words"] = len(tokens["words"])
            # Calcular comprimento médio das palavras
            features["avg_word_length"] = np.mean([len(w) for w in tokens["words"]]) if tokens["words"] else 0
            
        # Características baseadas em entidades (se disponíveis)
        if "entities" in tokens:
            entity_types = Counter(ent_type for _, ent_type in tokens["entities"])
            for ent_type, count in entity_types.items():
                features[f"entity_{ent_type}_count"] = count
            
            # Total de entidades
            features["total_entities"] = len(tokens["entities"])
        
        return features
    
    def _extract_metadata_features(self, preprocessed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrai características dos metadados.
        
        Args:
            preprocessed_data: Dados preprocessados
            
        Returns:
            Dicionário com características baseadas em metadados
        """
        features = {}
        metadata = preprocessed_data.get("metadata", {})
        
        # Fonte do documento
        if "source" in metadata:
            features["has_source"] = True
            source = str(metadata["source"]).lower()
            for key in ["site", "pdf", "email", "internal", "external", "report"]:
                features[f"source_contains_{key}"] = key in source
        else:
            features["has_source"] = False
            
        # Tipo de documento explícito
        if "document_type" in metadata:
            features["has_doc_type"] = True
            doc_type = str(metadata["document_type"]).lower()
            # Verificar correspondência com cada tipo conhecido
            for type_key in self.type_indicators.keys():
                features[f"metadata_type_{type_key}"] = type_key.lower() in doc_type
        else:
            features["has_doc_type"] = False
            
        # Data do documento
        if "date" in metadata:
            features["has_date"] = True
        else:
            features["has_date"] = False
            
        return features
    
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
        Extrai características binárias dos dados preprocessados.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Array numpy com características binárias
        """
        # Validar entrada
        is_valid, error = self._validate_input(preprocessed_data)
        if not is_valid:
            logger.error(f"Validação de entrada falhou: {error['error']}")
            if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
                # Retornar vetor de zeros como fallback
                return np.zeros((len(self.vectorizer.get_feature_names_out())))
            return np.array([])
        
        # Obter texto para processamento
        if "normalized_text" in preprocessed_data:
            text = preprocessed_data["normalized_text"]
        else:
            text = preprocessed_data["original_text"]
            
        # Verificar se o modelo está em modo de predição (com vectorizer treinado)
        if not self.vectorizer or not hasattr(self.vectorizer, 'vocabulary_'):
            # Em modo de extração para treinamento, retornamos o texto normalizado
            logger.debug("Modo de treinamento: retornando texto para vetorização posterior")
            return text
                
        # Em modo de predição, extrair características de texto e customizadas
        try:
            # Extrair características de texto
            text_features = self._extract_text_features(text)
            
            # Extrair características customizadas, se implementadas
            custom_features = self._extract_custom_features(preprocessed_data)
            
            # Combinar características
            combined_features = self._combine_features(text_features, custom_features)
            
            return combined_features
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {str(e)}")
            # Retornar vetor de zeros como fallback
            if self.vectorizer and hasattr(self.vectorizer, 'vocabulary_'):
                return np.zeros((len(self.vectorizer.get_feature_names_out())))
            return np.array([])
    
    def _prepare_training_features(self, training_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Sobrescreve o método da classe base para incluir vetorização de texto
        e extração de características personalizadas.
        
        Args:
            training_data: Lista de dados preprocessados
            
        Returns:
            Array numpy com as características para treinamento
        """
        start_time = time.time()
        logger.info(f"Preparando características para treinamento com {len(training_data)} amostras")
        
        # Extrair textos normalizados e características customizadas
        texts = []
        custom_features_list = []
        
        for data in training_data:
            # Validar entrada
            is_valid, error = self._validate_input(data)
            if not is_valid:
                logger.warning(f"Amostra inválida: {error['error']}")
                texts.append("")
                custom_features_list.append({})
                continue
                
            # Obter texto
            if "normalized_text" in data:
                text = data["normalized_text"]
            elif "original_text" in data:
                text = data["original_text"]
            else:
                logger.warning("Amostra sem texto, usando string vazia")
                text = ""
                
            texts.append(text)
            
            # Extrair características customizadas
            custom_features = self._extract_custom_features(data)
            custom_features_list.append(custom_features)
        
        # Log da configuração do vetorizador
        logger.info(f"Inicializando vetorizador com: max_features={self.max_features}, "
                   f"min_df={self.min_df}, ngram_range={self.ngram_range}")
            
        # Inicializar e treinar o vetorizador
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            binary=True  # Características binárias para Bernoulli NB
        )
        
        # Vetorizar textos
        logger.info("Vetorizando textos...")
        X_text = self.vectorizer.fit_transform(texts).toarray()
        
        # Armazenar nomes das características
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Log das características extraídas
        vocab_size = len(self.vectorizer.vocabulary_)
        logger.info(f"Vetorização concluída: {vocab_size} termos extraídos do vocabulário")
        
        if logger.isEnabledFor(logging.DEBUG):
            # Log dos top N termos mais frequentes
            top_n = min(20, vocab_size)
            word_counts = X_text.sum(axis=0)
            top_indices = word_counts.argsort()[-top_n:][::-1]
            top_terms = [(self.feature_names[i], int(word_counts[i])) for i in top_indices]
            logger.debug(f"Top {top_n} termos mais frequentes: {top_terms}")
        
        # Verificar se temos características customizadas para incorporar
        if custom_features_list and any(custom_features_list):
            logger.info("Processando características customizadas...")
            
            # Converter para DataFrame para manipulação mais fácil
            custom_df = pd.DataFrame(custom_features_list)
            
            # Selecionar apenas colunas numéricas e booleanas
            numeric_cols = custom_df.select_dtypes(include=['number', 'bool']).columns
            
            if not numeric_cols.empty:
                X_custom = custom_df[numeric_cols].values
                
                # Combinar características de texto e customizadas
                X_combined = np.hstack((X_text, X_custom))
                
                # Atualizar nomes das características
                self.feature_names = np.concatenate([self.feature_names, np.array(numeric_cols)])
                
                logger.info(f"Adicionadas {len(numeric_cols)} características customizadas")
                
                # Log do tempo de processamento
                processing_time = time.time() - start_time
                logger.info(f"Preparação de características concluída em {processing_time:.2f} segundos")
                
                # Dimensões do conjunto de treinamento
                logger.info(f"Dimensões finais do conjunto de características: {X_combined.shape}")
                
                return X_combined
            else:
                logger.warning("Nenhuma característica customizada numérica ou booleana encontrada")
        
        # Log do tempo de processamento
        processing_time = time.time() - start_time
        logger.info(f"Preparação de características concluída em {processing_time:.2f} segundos")
        
        # Se não há características customizadas utilizáveis, retornar apenas texto
        logger.info(f"Dimensões finais do conjunto de características: {X_text.shape}")
        return X_text
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sobrescreve o método da classe base para incluir extração de 
        características personalizadas na predição.
        
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
            
        start_time = time.time()
        logger.debug(f"Iniciando predição para amostra")
        
        try:
            # Extrair características
            features = self._extract_features(data)
            
            # Verificar dimensão das características
            if len(features.shape) == 1:
                # Reshapear para batch de uma amostra
                features = features.reshape(1, -1)
                
            # Verificar se o tamanho das características é compatível com o modelo
            expected_size = len(self.feature_names)
            actual_size = features.shape[1]
            
            if actual_size != expected_size:
                logger.warning(f"Dimensão incompatível de características: obtido {actual_size}, "
                              f"esperado {expected_size}")
                
                # Ajustar tamanho (preencher com zeros ou truncar)
                if actual_size < expected_size:
                    padded = np.zeros((1, expected_size))
                    padded[0, :actual_size] = features[0, :actual_size]
                    features = padded
                else:
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
            if predicted_class_str in self.class_mapping:
                readable_class = self.class_mapping[predicted_class_str]
            else:
                readable_class = predicted_class_str
            
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
            text = data.get("normalized_text", data.get("original_text", ""))
            top_terms = self._get_top_terms_for_class(predicted_class_str, text, top_n=5)
            
            # Resultado da predição
            result = {
                "predicted_class": predicted_class_str,
                "readable_class": readable_class,
                "probabilities": class_probs,
                "confidence": float(max(probabilities)),
                "inference_time": inference_time,
                "relevant_terms": top_terms
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
        Identifica os termos mais relevantes para a classificação.
        
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
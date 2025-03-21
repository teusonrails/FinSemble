"""
Módulo para gerenciamento eficiente de características textuais.

Este módulo implementa estratégias para extração, armazenamento e recuperação
eficientes de características textuais, otimizando o uso de memória.
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, Any, List, Union, Optional, Tuple
import hashlib
import logging
from functools import lru_cache

# Configuração de logging
logger = logging.getLogger(__name__)

class FeatureManager:
    """
    Gerenciador centralizado para extração e armazenamento eficiente de características.
    
    Esta classe implementa:
    1. Manutenção de matrizes esparsas quando apropriado
    2. Cache de características para textos frequentes
    3. Pipeline unificado de extração para diferentes classificadores
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o gerenciador de características.
        
        Args:
            config: Configurações para o gerenciador
                - cache_size: Tamanho do cache de características
                - keep_sparse: Manter representações esparsas quando possível
                - max_dense_features: Número máximo de características para conversão densa
        """
        self.config = config or {}
        self.cache_size = self.config.get("cache_size", 1024)
        self.keep_sparse = self.config.get("keep_sparse", True)
        self.max_dense_features = self.config.get("max_dense_features", 10000)
        
        # Dicionários para armazenar extractors e vectorizers
        self.vectorizers = {}
        self.feature_extractors = {}
        
        # Status e estatísticas
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "sparse_matrices": 0,
            "dense_matrices": 0
        }
        
        logger.info(f"FeatureManager inicializado com cache_size={self.cache_size}, "
                   f"keep_sparse={self.keep_sparse}")
    
    def register_vectorizer(self, name: str, vectorizer):
        """
        Registra um vectorizer para uso pelo gerenciador.
        
        Args:
            name: Nome do vectorizer
            vectorizer: Instância do vectorizer (ex: CountVectorizer, TfidfVectorizer)
        """
        self.vectorizers[name] = vectorizer
        logger.debug(f"Vectorizer '{name}' registrado")
    
    def register_extractor(self, name: str, extractor):
        """
        Registra um extrator de características personalizado.
        
        Args:
            name: Nome do extractor
            extractor: Função ou objeto para extração de características
        """
        self.feature_extractors[name] = extractor
        logger.debug(f"Extrator '{name}' registrado")
    
    @lru_cache(maxsize=1024)
    def _get_text_hash(self, text: str) -> str:
        """
        Calcula um hash único para um texto para uso no cache.
        
        Args:
            text: Texto para gerar hash
            
        Returns:
            String hash do texto
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def extract_text_features(self, text: str, vectorizer_name: str, 
                            as_sparse: Optional[bool] = None) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Extrai características textuais de forma eficiente.
        
        Args:
            text: Texto para extração
            vectorizer_name: Nome do vectorizer a utilizar
            as_sparse: Forçar retorno como matriz esparsa (None = usar configuração padrão)
            
        Returns:
            Características extraídas como array denso ou matriz esparsa
        """
        if vectorizer_name not in self.vectorizers:
            raise ValueError(f"Vectorizer '{vectorizer_name}' não registrado")
            
        vectorizer = self.vectorizers[vectorizer_name]
        
        # Verificar se o vectorizer está treinado
        if not hasattr(vectorizer, 'vocabulary_'):
            logger.warning(f"Vectorizer '{vectorizer_name}' não está treinado")
            return np.array([])
            
        # Determinar se deve manter esparso
        keep_sparse = as_sparse if as_sparse is not None else self.keep_sparse
        vocab_size = len(vectorizer.vocabulary_)
        
        if keep_sparse and vocab_size > self.max_dense_features:
            # Manter como matriz esparsa para eficiência
            features = vectorizer.transform([text])
            self.stats["sparse_matrices"] += 1
            return features
        else:
            # Converter para array denso
            features = vectorizer.transform([text]).toarray()[0]
            self.stats["dense_matrices"] += 1
            return features
    
    def extract_custom_features(self, data: Dict[str, Any], 
                              extractor_name: str) -> Dict[str, Any]:
        """
        Extrai características customizadas eficientemente com cache.
        
        Args:
            data: Dados para extração
            extractor_name: Nome do extrator a utilizar
            
        Returns:
            Dicionário com características extraídas
        """
        if extractor_name not in self.feature_extractors:
            raise ValueError(f"Extrator '{extractor_name}' não registrado")
            
        extractor = self.feature_extractors[extractor_name]
        
        # Para textos, podemos usar cache baseado em hash
        if "normalized_text" in data or "original_text" in data:
            text = data.get("normalized_text", data.get("original_text", ""))
            text_hash = self._get_text_hash(text)
            
            # Tentar obter do cache LRU decorado na função
            return self._cached_extraction(text_hash, extractor, data)
        else:
            # Para outros tipos de dados, extrair diretamente
            return extractor(data)
    
    @lru_cache(maxsize=1024)
    def _cached_extraction(self, text_hash: str, extractor, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Wrapper com cache para extração de características.
        
        Args:
            text_hash: Hash do texto para chave de cache
            extractor: Função de extração
            data: Dados para extração
            
        Returns:
            Características extraídas
        """
        self.stats["cache_hits"] += 1
        return extractor(data)
    
    def combine_features(self, 
                       text_features: Union[np.ndarray, sp.csr_matrix],
                       custom_features: Dict[str, Any],
                       force_dense: bool = True) -> Union[np.ndarray, sp.csr_matrix]:
        """
        Combina características textuais e customizadas de forma eficiente.
        
        Args:
            text_features: Características extraídas do texto
            custom_features: Características customizadas
            force_dense: Forçar conversão para array denso no resultado
            
        Returns:
            Características combinadas
        """
        if not custom_features:
            if force_dense and sp.issparse(text_features):
                return text_features.toarray()[0]
            return text_features
            
        # Converter características customizadas para array
        import pandas as pd
        custom_df = pd.DataFrame([custom_features])
        numeric_cols = custom_df.select_dtypes(include=['number', 'bool']).columns
        
        if not numeric_cols.empty:
            custom_array = custom_df[numeric_cols].values[0]
            
            # Combinar com características de texto
            if sp.issparse(text_features):
                if force_dense:
                    # Converter para denso e combinar
                    dense_text = text_features.toarray()[0]
                    return np.concatenate([dense_text, custom_array])
                else:
                    # Manter esparso (mais complexo - precisamos converter custom_array)
                    custom_sparse = sp.csr_matrix(custom_array.reshape(1, -1))
                    return sp.hstack([text_features, custom_sparse])
            else:
                # Ambos são densos
                return np.concatenate([text_features, custom_array])
        else:
            # Sem características customizadas numéricas
            if force_dense and sp.issparse(text_features):
                return text_features.toarray()[0]
            return text_features
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de uso do gerenciador.
        
        Returns:
            Dicionário com estatísticas
        """
        cache_total = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = self.stats["cache_hits"] / cache_total if cache_total > 0 else 0
        
        return {
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "sparse_matrices": self.stats["sparse_matrices"],
            "dense_matrices": self.stats["dense_matrices"]
        }
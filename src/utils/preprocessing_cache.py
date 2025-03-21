"""
Módulo para sistema de cache inteligente no preprocessamento.

Este módulo implementa mecanismos de cache para eliminar computações
redundantes durante o preprocessamento de textos financeiros.
"""

import hashlib
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from functools import lru_cache

# Configuração de logging
logger = logging.getLogger(__name__)

class PreprocessingCache:
    """
    Sistema de cache para operações de preprocessamento.
    
    Implementa um cache em vários níveis para diferentes etapas do
    preprocessamento, evitando computações redundantes.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o sistema de cache.
        
        Args:
            config: Configurações para o cache, incluindo:
                - cache_size: Tamanho máximo do cache por nível
                - enable_normalization_cache: Ativar cache para normalização
                - enable_tokenization_cache: Ativar cache para tokenização
                - enable_feature_cache: Ativar cache para extração de características
        """
        self.config = config or {}
        self.cache_size = self.config.get("cache_size", 1024)
        
        # Flags para ativar/desativar níveis específicos de cache
        self.enable_normalization_cache = self.config.get("enable_normalization_cache", True)
        self.enable_tokenization_cache = self.config.get("enable_tokenization_cache", True)
        self.enable_feature_cache = self.config.get("enable_feature_cache", True)
        
        # Caches para diferentes níveis de processamento
        self.normalization_cache = {}
        self.tokenization_cache = {}
        self.feature_cache = {}
        
        # Estatísticas para monitoramento
        self.stats = {
            "normalization_hits": 0,
            "normalization_misses": 0,
            "tokenization_hits": 0,
            "tokenization_misses": 0,
            "feature_hits": 0,
            "feature_misses": 0,
            "total_cache_hits": 0,
            "total_cache_misses": 0,
            "bytes_saved": 0
        }
        
        logger.info(f"Sistema de cache de preprocessamento inicializado com tamanho {self.cache_size}")
    
    def _get_text_hash(self, text: str) -> str:
        """
        Calcula um hash único para o texto.
        
        Args:
            text: Texto para gerar hash
            
        Returns:
            String hash MD5 do texto
        """
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_config_hash(self, config: Dict[str, Any]) -> str:
        """
        Calcula um hash para a configuração.
        
        Args:
            config: Configuração para gerar hash
            
        Returns:
            String hash da configuração
        """
        # Ordenar chaves para garantir consistência
        config_str = str(sorted(config.items()))
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_normalized_text(self, text: str, config: Dict[str, Any]) -> Optional[str]:
        """
        Tenta obter texto normalizado do cache.
        
        Args:
            text: Texto original
            config: Configuração de normalização
            
        Returns:
            Texto normalizado ou None se não estiver no cache
        """
        if not self.enable_normalization_cache:
            self.stats["normalization_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
            
        # Criar chave de cache combinando hash do texto e da configuração
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{config_hash}"
        
        if cache_key in self.normalization_cache:
            self.stats["normalization_hits"] += 1
            self.stats["total_cache_hits"] += 1
            self.stats["bytes_saved"] += len(text)
            return self.normalization_cache[cache_key]
        else:
            self.stats["normalization_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
    
    def cache_normalized_text(self, text: str, normalized_text: str, 
                            config: Dict[str, Any]) -> None:
        """
        Armazena texto normalizado no cache.
        
        Args:
            text: Texto original
            normalized_text: Texto normalizado
            config: Configuração de normalização
        """
        if not self.enable_normalization_cache:
            return
            
        # Manter o cache dentro do limite de tamanho
        if len(self.normalization_cache) >= self.cache_size:
            # Remover item mais antigo (FIFO)
            oldest_key = next(iter(self.normalization_cache))
            del self.normalization_cache[oldest_key]
            
        # Criar chave de cache e armazenar
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{config_hash}"
        
        self.normalization_cache[cache_key] = normalized_text
    
    def get_tokens(self, text: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Tenta obter tokens do cache.
        
        Args:
            text: Texto para tokenização
            config: Configuração de tokenização
            
        Returns:
            Tokens ou None se não estiver no cache
        """
        if not self.enable_tokenization_cache:
            self.stats["tokenization_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
            
        # Criar chave de cache
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{config_hash}"
        
        if cache_key in self.tokenization_cache:
            self.stats["tokenization_hits"] += 1
            self.stats["total_cache_hits"] += 1
            return self.tokenization_cache[cache_key]
        else:
            self.stats["tokenization_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
    
    def cache_tokens(self, text: str, tokens: Dict[str, Any], 
                   config: Dict[str, Any]) -> None:
        """
        Armazena tokens no cache.
        
        Args:
            text: Texto original ou normalizado
            tokens: Resultado da tokenização
            config: Configuração de tokenização
        """
        if not self.enable_tokenization_cache:
            return
            
        # Manter o cache dentro do limite de tamanho
        if len(self.tokenization_cache) >= self.cache_size:
            # Remover item mais antigo (FIFO)
            oldest_key = next(iter(self.tokenization_cache))
            del self.tokenization_cache[oldest_key]
            
        # Criar chave de cache e armazenar
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{config_hash}"
        
        self.tokenization_cache[cache_key] = tokens
    
    def get_features(self, text: str, feature_type: str, 
                   config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Tenta obter características do cache.
        
        Args:
            text: Texto normalizado
            feature_type: Tipo de características ('type', 'sentiment', 'impact', etc)
            config: Configuração de extração
            
        Returns:
            Características ou None se não estiver no cache
        """
        if not self.enable_feature_cache:
            self.stats["feature_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
            
        # Criar chave de cache
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{feature_type}_{config_hash}"
        
        if cache_key in self.feature_cache:
            self.stats["feature_hits"] += 1
            self.stats["total_cache_hits"] += 1
            return self.feature_cache[cache_key]
        else:
            self.stats["feature_misses"] += 1
            self.stats["total_cache_misses"] += 1
            return None
    
    def cache_features(self, text: str, feature_type: str, 
                     features: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Armazena características no cache.
        
        Args:
            text: Texto normalizado
            feature_type: Tipo de características
            features: Características extraídas
            config: Configuração de extração
        """
        if not self.enable_feature_cache:
            return
            
        # Manter o cache dentro do limite de tamanho
        if len(self.feature_cache) >= self.cache_size:
            # Remover item mais antigo (FIFO)
            oldest_key = next(iter(self.feature_cache))
            del self.feature_cache[oldest_key]
            
        # Criar chave de cache e armazenar
        text_hash = self._get_text_hash(text)
        config_hash = self._get_config_hash(config)
        cache_key = f"{text_hash}_{feature_type}_{config_hash}"
        
        self.feature_cache[cache_key] = features
    
    def clear_cache(self, level: Optional[str] = None) -> None:
        """
        Limpa o cache.
        
        Args:
            level: Nível específico para limpar ('normalization', 'tokenization', 'feature')
                  ou None para limpar todos
        """
        if level == "normalization" or level is None:
            self.normalization_cache.clear()
            
        if level == "tokenization" or level is None:
            self.tokenization_cache.clear()
            
        if level == "feature" or level is None:
            self.feature_cache.clear()
            
        logger.info(f"Cache de preprocessamento limpo (nível: {level or 'todos'})")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do cache.
        
        Returns:
            Dicionário com estatísticas de uso do cache
        """
        # Calcular taxas de acerto
        total_requests = self.stats["total_cache_hits"] + self.stats["total_cache_misses"]
        hit_rate = self.stats["total_cache_hits"] / total_requests if total_requests > 0 else 0
        
        norm_requests = self.stats["normalization_hits"] + self.stats["normalization_misses"]
        norm_hit_rate = self.stats["normalization_hits"] / norm_requests if norm_requests > 0 else 0
        
        token_requests = self.stats["tokenization_hits"] + self.stats["tokenization_misses"]
        token_hit_rate = self.stats["tokenization_hits"] / token_requests if token_requests > 0 else 0
        
        feature_requests = self.stats["feature_hits"] + self.stats["feature_misses"]
        feature_hit_rate = self.stats["feature_hits"] / feature_requests if feature_requests > 0 else 0
        
        return {
            "hit_rate": hit_rate,
            "normalization_hit_rate": norm_hit_rate,
            "tokenization_hit_rate": token_hit_rate,
            "feature_hit_rate": feature_hit_rate,
            "cache_size": {
                "normalization": len(self.normalization_cache),
                "tokenization": len(self.tokenization_cache),
                "feature": len(self.feature_cache),
            },
            "bytes_saved": self.stats["bytes_saved"],
            "total_hits": self.stats["total_cache_hits"],
            "total_misses": self.stats["total_cache_misses"]
        }
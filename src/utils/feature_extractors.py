"""
Módulo para extratores de características de texto do sistema FinSemble.

Este módulo contém classes especializadas para extração de diferentes tipos de 
características de texto, como léxicas, linguísticas e estruturais, utilizadas
pelos classificadores do sistema.
"""

import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Set
from collections import Counter
from abc import ABC, abstractmethod

# Configuração de logging
logger = logging.getLogger(__name__)

class BaseFeatureExtractor(ABC):
    """Classe base para todos os extratores de características."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator de características.
        
        Args:
            config: Configurações para o extrator
        """
        self.config = config or {}
    
    @abstractmethod
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai características do texto.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais específicos do extrator
            
        Returns:
            Dicionário com características extraídas
        """
        pass
    
    def _validate_text(self, text: str) -> str:
        """
        Valida e formata o texto de entrada.
        
        Args:
            text: Texto para validação
            
        Returns:
            Texto validado e formatado
        """
        if not text or not isinstance(text, str):
            logger.warning(f"Texto inválido para extração de características: {text}")
            return ""
        return text.lower().strip()

class LexiconFeatureExtractor(BaseFeatureExtractor):
    """Extrator de características baseadas em léxicos de sentimento."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator de características léxicas.
        
        Args:
            config: Configurações incluindo léxicos de sentimento
        """
        super().__init__(config)
        self.sentiment_lexicon = config.get("sentiment_lexicon", {})
        self.negation_terms = config.get("negation_terms", [])
        self.intensifiers = config.get("intensifiers", [])
        self.diminishers = config.get("diminishers", [])
        
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai características baseadas em léxico de sentimento.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com características léxicas
        """
        text = self._validate_text(text)
        if not text:
            return {}
            
        features = {}
        
        # Contar termos de cada categoria de sentimento
        for sentiment, terms in self.sentiment_lexicon.items():
            count = sum(text.count(term.lower()) for term in terms)
            features[f"{sentiment}_terms_count"] = count
            
            # Calcular densidade de termos (normalizada pelo comprimento do texto)
            word_count = len(text.split())
            if word_count > 0:
                features[f"{sentiment}_terms_density"] = count / word_count
            else:
                features[f"{sentiment}_terms_density"] = 0
                
        # Detectar negações e seu impacto
        negation_count = sum(text.count(term) for term in self.negation_terms)
        features["negation_count"] = negation_count
        
        # Detectar intensificadores e atenuadores
        intensifier_count = sum(text.count(term) for term in self.intensifiers)
        diminisher_count = sum(text.count(term) for term in self.diminishers)
        features["intensifier_count"] = intensifier_count
        features["diminisher_count"] = diminisher_count
        
        # Análise de padrões de negação que podem inverter o sentimento
        negation_patterns = []
        for negation in self.negation_terms:
            # Regex para capturar frases com negação seguida de até 5 palavras
            pattern = fr'{negation}\s+(\w+\s+){{0,5}}(\w+)'
            negation_patterns.extend(re.findall(pattern, text))
        features["negation_patterns_count"] = len(negation_patterns)
        
        return features

class StructuralFeatureExtractor(BaseFeatureExtractor):
    """Extrator de características estruturais do texto."""
    
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai características estruturais do texto.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais incluindo tokens
            
        Returns:
            Dicionário com características estruturais
        """
        features = {}
        tokens = kwargs.get("tokens", {})
        
        # Características baseadas em sentenças
        if "sentences" in tokens:
            features["num_sentences"] = len(tokens["sentences"])
            sent_lengths = [len(s.split()) for s in tokens["sentences"]]
            features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0
            features["var_sentence_length"] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
            
        # Características baseadas em palavras
        if "words" in tokens:
            features["num_words"] = len(tokens["words"])
            features["avg_word_length"] = np.mean([len(w) for w in tokens["words"]]) if tokens["words"] else 0
            
        # Características baseadas em entidades
        if "entities" in tokens:
            entity_types = Counter(ent_type for _, ent_type in tokens["entities"])
            for ent_type, count in entity_types.items():
                features[f"entity_{ent_type}_count"] = count
            features["total_entities"] = len(tokens["entities"])
            
        return features

class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extrator de características temporais do texto."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator de características temporais.
        
        Args:
            config: Configurações incluindo indicadores temporais
        """
        super().__init__(config)
        self.future_indicators = config.get("future_indicators", [
            "irá", "deverá", "prevê", "projeta", "espera", "antecipa", "planeja"
        ])
        self.past_indicators = config.get("past_indicators", [
            "foi", "registrou", "apresentou", "obteve", "alcançou", "realizou"
        ])
        
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai características temporais do texto.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com características temporais
        """
        text = self._validate_text(text)
        if not text:
            return {}
            
        features = {}
        
        # Analisar orientação temporal
        features["future_orientation_count"] = sum(text.count(word) for word in self.future_indicators)
        features["past_orientation_count"] = sum(text.count(word) for word in self.past_indicators)
        features["temporal_ratio"] = (
            features["future_orientation_count"] / features["past_orientation_count"]
            if features["past_orientation_count"] > 0 else 
            float('inf') if features["future_orientation_count"] > 0 else 1.0
        )
        
        return features

class CompositeFeatureExtractor:
    """Combina múltiplos extratores de características."""
    
    def __init__(self, extractors: List[BaseFeatureExtractor]):
        """
        Inicializa o extrator composto.
        
        Args:
            extractors: Lista de extratores de características
        """
        self.extractors = extractors
        
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai características usando todos os extratores registrados.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário combinado com todas as características
        """
        all_features = {}
        
        for extractor in self.extractors:
            try:
                features = extractor.extract(text, **kwargs)
                all_features.update(features)
            except Exception as e:
                logger.error(f"Erro ao extrair características com {extractor.__class__.__name__}: {str(e)}")
                
        return all_features
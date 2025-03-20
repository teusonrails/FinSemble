"""
Módulo para extração de características numéricas de textos financeiros.

Este módulo contém classes e funções especializadas para extrair valores
numéricos, percentuais, monetários e outras características quantitativas
de textos financeiros.
"""
import re
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# Configuração de logging
logger = logging.getLogger(__name__)

class BaseExtractor(ABC):
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


class NumericalExtractor(BaseExtractor):
    """Extrator de valores numéricos, percentuais e monetários."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator numérico.
        
        Args:
            config: Configurações para o extrator
        """
        super().__init__(config)
        
        # Padrões de expressão regular para extração
        self.percentage_pattern = r'(\d+(?:[.,]\d+)?)(?:\s*)?%'
        self.currency_pattern = r'R\$\s*(\d+(?:[.,]\d+)?(?:\s*\w+)?)|(\d+(?:[.,]\d+)?(?:\s*\w+)?\s*reais)|(\$\s*\d+(?:[.,]\d+)?)'
        self.numeric_value_pattern = r'\b(\d+(?:[.,]\d+)?)\s*(milhões|bilhões|mil|mi|bi)?\b'
        
        # Configurar limiares para categorização de percentuais
        self.high_percentage_threshold = config.get("high_percentage_threshold", 15)
        self.medium_percentage_threshold = config.get("medium_percentage_threshold", 5)
    
    def extract(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Extrai valores numéricos do texto.
        
        Args:
            text: Texto para extração
            **kwargs: Argumentos adicionais
            
        Returns:
            Dicionário com características numéricas
        """
        text = self._validate_text(text)
        if not text:
            return self._get_empty_features()
        
        features = {}
        
        # Extração de percentuais
        percentages = [float(p.replace(',', '.')) for p in re.findall(self.percentage_pattern, text)]
        features["percentage_count"] = len(percentages)
        
        if percentages:
            features["max_percentage"] = max(percentages)
            features["min_percentage"] = min(percentages)
            features["avg_percentage"] = sum(percentages) / len(percentages)
            features["sum_percentage"] = sum(percentages)
            
            # Categorizar percentuais por magnitude
            features["high_percentage_count"] = sum(1 for p in percentages if p >= self.high_percentage_threshold)
            features["medium_percentage_count"] = sum(1 for p in percentages if self.medium_percentage_threshold <= p < self.high_percentage_threshold)
            features["low_percentage_count"] = sum(1 for p in percentages if p < self.medium_percentage_threshold)
        else:
            # Valores padrão quando não há percentuais
            features.update(self._get_empty_percentage_features())
        
        # Extração de valores monetários
        currency_matches = re.findall(self.currency_pattern, text)
        features["currency_count"] = len(currency_matches)
        
        # Extração de valores numéricos genéricos
        numeric_values = re.findall(self.numeric_value_pattern, text)
        features["numeric_value_count"] = len(numeric_values)
        
        # Calcular densidade de valores numéricos no texto
        word_count = len(text.split())
        if word_count > 0:
            total_numerical = features["percentage_count"] + features["currency_count"] + features["numeric_value_count"]
            features["numerical_density"] = total_numerical / word_count
        else:
            features["numerical_density"] = 0
            
        return features
    
    def _get_empty_features(self) -> Dict[str, Any]:
        """
        Retorna características vazias para quando não há texto válido.
        
        Returns:
            Dicionário com características vazias
        """
        features = {
            "percentage_count": 0,
            "currency_count": 0,
            "numeric_value_count": 0,
            "numerical_density": 0
        }
        features.update(self._get_empty_percentage_features())
        return features
    
    def _get_empty_percentage_features(self) -> Dict[str, Any]:
        """
        Retorna características vazias relacionadas a percentuais.
        
        Returns:
            Dicionário com características de percentuais vazias
        """
        return {
            "max_percentage": 0,
            "min_percentage": 0,
            "avg_percentage": 0,
            "sum_percentage": 0,
            "high_percentage_count": 0,
            "medium_percentage_count": 0,
            "low_percentage_count": 0
        }


class StructuralExtractor(BaseExtractor):
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
        
        # Obter tokens do preprocessador
        tokens = kwargs.get("tokens", {})
        
        # Características estruturais básicas
        if "sentences" in tokens:
            features["num_sentences"] = len(tokens["sentences"])
            
            # Calcular comprimento médio e variância das sentenças
            sent_lengths = [len(s.split()) for s in tokens["sentences"]]
            features["avg_sentence_length"] = np.mean(sent_lengths) if sent_lengths else 0
            features["var_sentence_length"] = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
            features["max_sentence_length"] = max(sent_lengths) if sent_lengths else 0
        else:
            features["num_sentences"] = 0
            features["avg_sentence_length"] = 0
            features["var_sentence_length"] = 0
            features["max_sentence_length"] = 0
            
        if "words" in tokens:
            features["num_words"] = len(tokens["words"])
            
            # Calcular comprimento médio das palavras
            word_lengths = [len(w) for w in tokens["words"]]
            features["avg_word_length"] = np.mean(word_lengths) if word_lengths else 0
        else:
            features["num_words"] = 0
            features["avg_word_length"] = 0
            
        # Características baseadas em entidades (se disponíveis)
        if "entities" in tokens:
            from collections import Counter
            
            # Contagem total de entidades
            features["total_entity_count"] = len(tokens["entities"])
            
            # Contagem de entidades por tipo
            entity_types = Counter(ent_type for _, ent_type in tokens["entities"])
            
            # Entidades financeiras específicas
            financial_entities = ["ORG", "MONEY", "PERCENT", "CARDINAL", "DATE"]
            for ent_type in financial_entities:
                features[f"entity_{ent_type}_count"] = entity_types.get(ent_type, 0)
                
            # Densidade de entidades financeiras
            if features["num_words"] > 0:
                financial_entity_count = sum(entity_types.get(ent_type, 0) for ent_type in financial_entities)
                features["financial_entity_density"] = financial_entity_count / features["num_words"]
            else:
                features["financial_entity_density"] = 0
        else:
            features["total_entity_count"] = 0
            features["financial_entity_density"] = 0
            for ent_type in ["ORG", "MONEY", "PERCENT", "CARDINAL", "DATE"]:
                features[f"entity_{ent_type}_count"] = 0
                
        return features


class CompositeExtractor:
    """Combina múltiplos extratores em um único pipeline."""
    
    def __init__(self, extractors: List[BaseExtractor]):
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
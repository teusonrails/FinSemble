"""
Módulo de utilitários para análise de impacto em textos financeiros.

Este módulo contém classes e funções para identificação de termos de impacto,
análise temporal e geração de explicações para predições de impacto.
"""
import logging
import random
from typing import Dict, List, Any, Optional
from collections import Counter

# Configuração de logging
logger = logging.getLogger(__name__)

class ImpactTermExtractor:
    """Extrator de termos de impacto do texto."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o extrator de termos de impacto.
        
        Args:
            config: Configurações para o extrator
        """
        self.config = config or {}
        self.impact_terms = self._initialize_impact_terms(config)
    
    def _initialize_impact_terms(self, config: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Inicializa os termos de impacto para diferentes categorias.
        
        Args:
            config: Configuração contendo possíveis termos de impacto customizados
            
        Returns:
            Dicionário com termos de impacto por categoria
        """
        # Usar termos definidos na configuração, se existirem
        if "impact_terms" in config:
            return config["impact_terms"]
        
        # Termos de impacto padrão
        return {
            "high_impact": [
                "significativo", "expressivo", "substancial", "considerável", "relevante",
                "forte", "intenso", "acentuado", "pronunciado", "profundo", "massivo",
                "radical", "drástico", "grande", "enorme", "extraordinário", "excepcional"
            ],
            "medium_impact": [
                "moderado", "mediano", "intermediário", "razoável", "parcial",
                "médio", "regular", "moderadamente", "considerável", "notável",
                "sensível", "importante", "apreciável", "perceptível"
            ],
            "low_impact": [
                "leve", "pequeno", "limitado", "suave", "sutil", "modesto", "ligeiro", 
                "tênue", "discreto", "mínimo", "marginal", "menor", "pouco", "reduzido",
                "insignificante", "desprezível", "irrisório"
            ]
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extrai características baseadas em termos de impacto.
        
        Args:
            text: Texto para extração
            
        Returns:
            Dicionário com características de impacto
        """
        text_lower = text.lower()
        features = {}
        
        # Contar termos de cada categoria de impacto
        for impact_category, terms in self.impact_terms.items():
            count = sum(text_lower.count(term.lower()) for term in terms)
            features[f"{impact_category}_count"] = count
            
            # Calcular densidade de termos (normalizada pelo comprimento do texto)
            word_count = len(text_lower.split())
            if word_count > 0:
                features[f"{impact_category}_density"] = count / word_count
            else:
                features[f"{impact_category}_density"] = 0
        
        # Calcular razões entre categorias
        if features["low_impact_count"] > 0:
            features["high_to_low_ratio"] = features["high_impact_count"] / features["low_impact_count"]
        else:
            features["high_to_low_ratio"] = features["high_impact_count"] if features["high_impact_count"] > 0 else 0
            
        if features["medium_impact_count"] > 0:
            features["high_to_medium_ratio"] = features["high_impact_count"] / features["medium_impact_count"]
        else:
            features["high_to_medium_ratio"] = features["high_impact_count"] if features["high_impact_count"] > 0 else 0
            
        return features


class TemporalAnalyzer:
    """Analisador de orientação temporal do texto."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o analisador temporal.
        
        Args:
            config: Configurações para o analisador
        """
        self.config = config or {}
        self.time_indicators = self._initialize_time_indicators()
    
    def _initialize_time_indicators(self) -> Dict[str, List[str]]:
        """
        Inicializa indicadores temporais.
        
        Returns:
            Dicionário com indicadores temporais por categoria
        """
        return {
            "short_term": [
                "imediato", "curto prazo", "semana", "mês", "trimestre", "próximos dias",
                "atual", "recente", "iminente", "breve"
            ],
            "medium_term": [
                "médio prazo", "semestre", "próximos meses", "ano corrente", "ano fiscal"
            ],
            "long_term": [
                "longo prazo", "anos", "década", "futuro", "estratégico", "estrutural",
                "permanente", "duradouro", "sustentável"
            ]
        }
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extrai características relacionadas à dimensão temporal.
        
        Args:
            text: Texto para extração
            
        Returns:
            Dicionário com características temporais
        """
        text_lower = text.lower()
        features = {}
        
        # Contar indicadores temporais
        for time_category, indicators in self.time_indicators.items():
            count = sum(text_lower.count(indicator.lower()) for indicator in indicators)
            features[f"{time_category}_count"] = count
        
        # Calcular orientação temporal predominante
        temporal_counts = [
            (features["short_term_count"], "short"),
            (features["medium_term_count"], "medium"),
            (features["long_term_count"], "long")
        ]
        
        predominant_timeframe = max(temporal_counts)[1] if any(count > 0 for count, _ in temporal_counts) else "none"
        features["predominant_timeframe"] = predominant_timeframe
        
        # Codificar a orientação temporal como valor numérico (curto=1, médio=2, longo=3, nenhum=0)
        timeframe_mapping = {"short": 1, "medium": 2, "long": 3, "none": 0}
        features["timeframe_numeric"] = timeframe_mapping[predominant_timeframe]
        
        return features


class CompositeScoreCalculator:
    """Calculador de pontuações compostas de impacto."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o calculador de pontuações compostas.
        
        Args:
            config: Configurações para o calculador
        """
        self.config = config or {}
        
        # Configurações de pesos
        self.percentage_weight = config.get("percentage_weight", 0.4)
        self.impact_terms_weight = config.get("impact_terms_weight", 0.4)
        self.temporal_weight = config.get("temporal_weight", 0.2)
    
    def calculate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula características compostas a partir das características extraídas.
        
        Args:
            features: Dicionário com características extraídas
            
        Returns:
            Dicionário com características compostas
        """
        composite_features = {}
        
        # Pontuação de impacto composta (combinação das múltiplas dimensões de impacto)
        # Esta é calculada considerando:
        # 1. Valores percentuais e sua magnitude
        # 2. Presença de termos de alto impacto
        # 3. Orientação temporal (curto prazo tem maior impacto imediato)
        
        # 1. Componente de percentuais
        percentage_score = (
            features.get("high_percentage_count", 0) * 1.5 +
            features.get("medium_percentage_count", 0) * 1.0 +
            features.get("low_percentage_count", 0) * 0.5
        ) / 10.0  # Normalizar para faixa 0-1 aproximada
        
        # Limitar a 1.0
        percentage_score = min(percentage_score, 1.0)
        
        # 2. Componente de termos de impacto
        impact_terms_score = (
            features.get("high_impact_density", 0) * 1.5 +
            features.get("medium_impact_density", 0) * 1.0 +
            features.get("low_impact_density", 0) * 0.5
        ) * 10.0  # Escalar para faixa 0-1 aproximada
        
        # Limitar a 1.0
        impact_terms_score = min(impact_terms_score, 1.0)
        
        # 3. Componente temporal (inverso do timeframe_numeric - curto prazo tem maior impacto imediato)
        timeframe_numeric = features.get("timeframe_numeric", 0)
        if timeframe_numeric > 0:
            # Inverter: 1=curto prazo (valor mais alto), 3=longo prazo (valor mais baixo)
            temporal_score = (4 - timeframe_numeric) / 3.0  # Normalizar para 0-1
        else:
            # Quando não há indicação temporal, usar valor neutro
            temporal_score = 0.5
            
        # Calcular pontuação composta
        composite_score = (
            percentage_score * self.percentage_weight +
            impact_terms_score * self.impact_terms_weight +
            temporal_score * self.temporal_weight
        )
        
        composite_features["percentage_impact_score"] = percentage_score
        composite_features["terms_impact_score"] = impact_terms_score
        composite_features["temporal_impact_score"] = temporal_score
        composite_features["composite_impact_score"] = composite_score
        
        # Índice de volatilidade - texto com muitos extremos (altos e baixos) pode indicar volatilidade
        high_impact = features.get("high_impact_count", 0)
        low_impact = features.get("low_impact_count", 0)
        if high_impact > 0 and low_impact > 0:
            composite_features["volatility_index"] = (high_impact * low_impact) / (high_impact + low_impact)
        else:
            composite_features["volatility_index"] = 0
            
        return composite_features


class ImpactExplainer:
    """Gerador de explicações para análise de impacto."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o explicador de impacto.
        
        Args:
            config: Configurações para o explicador
        """
        self.config = config or {}
        self.class_mapping = config.get("class_mapping", {
            "high": "Alto Impacto",
            "medium": "Médio Impacto",
            "low": "Baixo Impacto"
        })
    
    def generate_summary(self, prediction: Dict[str, Any], features: Dict[str, Any]) -> str:
        """
        Gera um resumo em linguagem natural do impacto previsto.
        
        Args:
            prediction: Resultado da predição
            features: Características extraídas
            
        Returns:
            Resumo em texto do impacto
        """
        impact = prediction["readable_class"]
        confidence = prediction["confidence"]
        
        # Classificar o nível de confiança
        confidence_level = "alta" if confidence > 0.8 else "moderada" if confidence > 0.6 else "baixa"
        
        # Termos de explicação por categoria de impacto
        impact_descriptions = {
            "Alto Impacto": [
                "significativo", "expressivo", "substancial", "considerável",
                "forte", "acentuado", "pronunciado"
            ],
            "Médio Impacto": [
                "moderado", "mediano", "intermediário", "razoável",
                "considerável", "apreciável", "notável"
            ],
            "Baixo Impacto": [
                "limitado", "reduzido", "pequeno", "leve",
                "sutil", "modesto", "discreto", "mínimo"
            ]
        }
        
        # Escolher um termo aleatório para variar a descrição
        if impact in impact_descriptions:
            impact_term = random.choice(impact_descriptions[impact])
        else:
            impact_term = "notável"
            
        # Gerar texto base com base no impacto
        summary = f"O texto apresenta um impacto {impact_term} (categoria: {impact}, confiança {confidence_level}: {confidence:.1%})"
        
        # Adicionar detalhes sobre os componentes da previsão
        components = []
        
        # Verificar se temos informações suficientes sobre componentes
        if "composite_impact_score" in features:
            # Componente percentual
            if features.get("percentage_count", 0) > 0:
                components.append(f"presença de {features['percentage_count']} valores percentuais")
                
            # Componente de termos de impacto
            high_count = features.get("high_impact_count", 0)
            if high_count > 0:
                components.append(f"{high_count} termos indicativos de alto impacto")
                
            # Componente temporal
            timeframe = features.get("predominant_timeframe", "none")
            if timeframe != "none":
                timeframe_terms = {
                    "short": "curto prazo",
                    "medium": "médio prazo",
                    "long": "longo prazo"
                }
                components.append(f"orientação temporal de {timeframe_terms.get(timeframe, timeframe)}")
                
        if components:
            summary += ". Fatores contribuintes: " + ", ".join(components)
                
        return summary
    
    def interpret_confidence_margin(self, margin: float) -> str:
        """
        Interpreta a margem de confiança entre as duas principais classes.
        
        Args:
            margin: Diferença entre as probabilidades das duas principais classes
            
        Returns:
            Interpretação em texto da margem
        """
        if margin > 0.5:
            return "muito alta (classificação inequívoca)"
        elif margin > 0.3:
            return "alta (classificação confiável)"
        elif margin > 0.15:
            return "moderada (classificação provável)"
        elif margin > 0.05:
            return "baixa (classificação incerta)"
        else:
            return "muito baixa (classificação ambígua)"
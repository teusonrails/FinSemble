"""
Módulo para modelagem de impacto de textos financeiros.

Este módulo implementa o Modelador de Impacto do FinSemble,
utilizando o algoritmo Gaussian Naive Bayes aprimorado para prever
o impacto quantitativo (alto, médio, baixo) de comunicações financeiras.
"""
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import os

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

from src.classifiers.base import BaseClassifier
from src.utils.numerical_extractors import (
    NumericalExtractor, 
    StructuralExtractor, 
    CompositeExtractor
)
from src.utils.impact_utils import (
    ImpactTermExtractor, 
    TemporalAnalyzer, 
    CompositeScoreCalculator, 
    ImpactExplainer
)
from src.utils.model_calibration import ModelCalibrator, EnhancedGaussianNB

# Configuração de logging
logger = logging.getLogger(__name__)

class ImpactModeler(BaseClassifier):
    """
    Modelador de impacto para textos financeiros.
    
    Utiliza o algoritmo Gaussian Naive Bayes aprimorado para classificar documentos
    com base em características numéricas e textuais para determinar o impacto
    quantitativo (alto, médio, baixo).
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o modelador de impacto.
        
        Args:
            config: Configurações específicas para o classificador, incluindo:
                - var_smoothing: Parâmetro de suavização para Gaussian NB
                - normalize: Se deve aplicar normalização às características
                - select_features: Se deve realizar seleção de características
                - n_features_to_select: Número de características a selecionar
                - class_mapping: Mapeamento entre códigos de classe e nomes legíveis
                - impact_thresholds: Limiares para categorias de impacto
                - run_calibration: Se deve executar calibração de parâmetros
        """
        # Chamar inicializador da classe base
        super().__init__(config)
        
        # Configurações específicas para Gaussian NB
        self.var_smoothing = config.get("var_smoothing", 1e-9)
        self.normalize = config.get("normalize", True)
        
        # Configurações para seleção de características
        self.select_features = config.get("select_features", False)
        self.n_features_to_select = config.get("n_features_to_select", 20)
        
        # Inicializar pipeline com componentes
        self.pipeline = None
        self.feature_selector = None
        
        # Mapeamento de classes para nomes legíveis
        self.class_mapping = config.get("class_mapping", {
            "high": "Alto Impacto",
            "medium": "Médio Impacto",
            "low": "Baixo Impacto"
        })
        
        # Limiares para categorias de impacto (serão refinados durante a calibração)
        self.impact_thresholds = config.get("impact_thresholds", {
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1
        })
        
        # Parâmetros de calibração
        self.run_calibration = config.get("run_calibration", False)
        self.calibration_folds = config.get("calibration_folds", 5)
        
        # Extratores de características
        self._initialize_feature_extractors(config)
        
        # Calibrador de modelo
        self.model_calibrator = ModelCalibrator({
            "calibration_folds": self.calibration_folds,
            "n_jobs": config.get("n_jobs", -1)
        })
        
        # Explicador de impacto
        self.impact_explainer = ImpactExplainer({
            "class_mapping": self.class_mapping
        })
        
        # Métricas de performance
        self.parallel_processing = config.get("parallel_processing", False)
        self.n_jobs = config.get("n_jobs", -1)  # -1 indica usar todos os núcleos disponíveis
        
        logger.info(f"Modelador de Impacto inicializado com parâmetros: "
                   f"var_smoothing={self.var_smoothing}, normalize={self.normalize}, "
                   f"select_features={self.select_features}")
    
    def _initialize_feature_extractors(self, config: Dict[str, Any]) -> None:
        """
        Inicializa os extratores de características.
        
        Args:
            config: Configurações para os extratores
        """
        # Inicializar extratores individuais
        self.numerical_extractor = NumericalExtractor(config)
        self.structural_extractor = StructuralExtractor()
        self.impact_term_extractor = ImpactTermExtractor(config)
        self.temporal_analyzer = TemporalAnalyzer(config)
        
        # Criar extrator composto para características básicas
        self.feature_extractor = CompositeExtractor([
            self.numerical_extractor,
            self.structural_extractor,
            self.impact_term_extractor,
            self.temporal_analyzer
        ])
        
        # Calculador de pontuações compostas
        self.composite_calculator = CompositeScoreCalculator(config)
    
    def _create_model(self) -> BaseEstimator:
        """
        Cria e configura o modelo Gaussian Naive Bayes aprimorado.
        
        Returns:
            Instância configurada do modelo na forma de um pipeline
        """
        logger.info(f"Criando modelo Gaussian NB com parâmetros: var_smoothing={self.var_smoothing}, "
                   f"normalize={self.normalize}")
        
        # Criar o modelo base
        model = EnhancedGaussianNB(
            var_smoothing=self.var_smoothing,
            normalize=self.normalize
        )
        
        # Construir pipeline de acordo com a configuração
        pipeline_steps = []
        
        # Adicionar seleção de características, se configurada
        if self.select_features:
            self.feature_selector = SelectKBest(f_classif, k=self.n_features_to_select)
            pipeline_steps.append(('feature_selection', self.feature_selector))
        
        # Adicionar o modelo final
        pipeline_steps.append(('classifier', model))
        
        # Criar pipeline
        self.pipeline = Pipeline(pipeline_steps)
        
        return self.pipeline
    
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
    
    def _extract_features(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extrai todas as características para modelagem de impacto.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Array numpy com as características extraídas
        """
        # Validar entrada
        is_valid, error = self._validate_input(preprocessed_data)
        if not is_valid:
            logger.error(f"Validação de entrada falhou: {error['error']}")
            # Retornar vetor de zeros como fallback
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                return np.zeros(len(self.feature_names_))
            return np.array([])
        
        # Obter texto para processamento
        if "normalized_text" in preprocessed_data:
            text = preprocessed_data["normalized_text"]
        else:
            text = preprocessed_data["original_text"]
            
        # Implementar a extração de características em paralelo se configurado
        if self.parallel_processing:
            return self._extract_features_parallel(text, preprocessed_data)
        else:
            return self._extract_features_sequential(text, preprocessed_data)
            
    def _extract_features_sequential(self, text: str, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Implementação sequencial de extração de características.
        
        Args:
            text: Texto para processamento
            preprocessed_data: Dados preprocessados
            
        Returns:
            Array numpy com as características extraídas
        """
        try:
            # Extrair características básicas com o extrator composto
            all_features = self.feature_extractor.extract(text, tokens=preprocessed_data.get("tokens", {}))
            
            # Calcular características compostas
            composite_features = self.composite_calculator.calculate(all_features)
            all_features.update(composite_features)
            
            # Converter para array numpy
            feature_df = pd.DataFrame([all_features])
            
            # Selecionar apenas colunas numéricas para classificação
            numeric_cols = feature_df.select_dtypes(include=['number']).columns
            features_array = feature_df[numeric_cols].values[0]
            
            # Armazenar nomes das características se estivermos em modo de treinamento
            if not hasattr(self, 'feature_names_') or self.feature_names_ is None:
                self.feature_names_ = np.array(numeric_cols)
                logger.debug(f"Armazenados {len(self.feature_names_)} nomes de características")
                
            return features_array
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {str(e)}", exc_info=True)
            # Retornar vetor de zeros como fallback
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                return np.zeros(len(self.feature_names_))
            return np.array([])
            
    def _extract_features_parallel(self, text: str, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Implementação paralela de extração de características.
        
        Args:
            text: Texto para processamento
            preprocessed_data: Dados preprocessados
            
        Returns:
            Array numpy com as características extraídas
        """
        try:
            # Definir as tarefas a serem executadas em paralelo
            tasks = [
                (self.numerical_extractor.extract, (text,)),
                (self.structural_extractor.extract, (text, {"tokens": preprocessed_data.get("tokens", {})})),
                (self.impact_term_extractor.extract, (text,)),
                (self.temporal_analyzer.extract, (text,))
            ]
            
            # Número de workers
            n_workers = min(len(tasks), self.n_jobs if self.n_jobs > 0 else os.cpu_count() or 4)
            
            # Executar tarefas em paralelo
            results = []
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(func, *args) for func, args in tasks]
                results = [future.result() for future in futures]
                
            # Combinar resultados
            all_features = {}
            for result in results:
                all_features.update(result)
                
            # Calcular características compostas (não paralelizado devido a dependências)
            composite_features = self.composite_calculator.calculate(all_features)
            all_features.update(composite_features)
            
            # Converter para array numpy
            feature_df = pd.DataFrame([all_features])
            
            # Selecionar apenas colunas numéricas para classificação
            numeric_cols = feature_df.select_dtypes(include=['number']).columns
            features_array = feature_df[numeric_cols].values[0]
            
            # Armazenar nomes das características se estivermos em modo de treinamento
            if not hasattr(self, 'feature_names_') or self.feature_names_ is None:
                self.feature_names_ = np.array(numeric_cols)
                logger.debug(f"Armazenados {len(self.feature_names_)} nomes de características em paralelo")
                
            return features_array
            
        except Exception as e:
            logger.error(f"Erro na extração paralela de características: {str(e)}", exc_info=True)
            # Retornar vetor de zeros como fallback
            if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                return np.zeros(len(self.feature_names_))
            return np.array([])
    
    def _prepare_training_features(self, training_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prepara características para treinamento, extraindo-as de cada amostra.
        
        Args:
            training_data: Lista de dados preprocessados
            
        Returns:
            Array numpy com características para treinamento
        """
        start_time = time.time()
        logger.info(f"Preparando características para treinamento com {len(training_data)} amostras")
        
        features_list = []
        
        for data in training_data:
            features = self._extract_features(data)
            features_list.append(features)
            
        # Converter lista de arrays para matriz
        features_matrix = np.array(features_list)
        
        processing_time = time.time() - start_time
        logger.info(f"Preparação de características concluída em {processing_time:.2f} segundos")
        logger.info(f"Dimensão da matriz de características: {features_matrix.shape}")
        
        return features_matrix
    
    def train(self, training_data: List[Dict[str, Any]], labels: List[Any], 
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Treina o modelador de impacto com os dados fornecidos.
        
        Args:
            training_data: Lista de dados preprocessados para treinamento
            labels: Lista de rótulos correspondentes
            validation_split: Proporção de dados para validação
            
        Returns:
            Resultados do treinamento
        """
        # Implementar a calibração de parâmetros e chamar o treinamento da classe base
        result = super().train(training_data, labels, validation_split)
        
        # Se a calibração estiver habilitada e o treinamento foi bem-sucedido, calibrar parâmetros
        if self.run_calibration and "success" in result and result["success"]:
            logger.info("Executando calibração de parâmetros após treinamento inicial...")
            
            # Extrair características de treinamento novamente
            X = self._prepare_training_features(training_data)
            y = np.array(labels)
            
            # Calibrar parâmetros do modelo
            param_grid = {
                'classifier__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
            }
            
            if self.select_features:
                param_grid['feature_selection__k'] = [5, 10, 15, 20, 25, 30]
                
            calibration_results = self.model_calibrator.calibrate_model_parameters(
                self.pipeline, X, y, param_grid
            )
            
            # Atualizar pipeline com os melhores parâmetros
            self.pipeline = calibration_results["best_estimator"]
            
            # Calibrar limiares de impacto
            calibrated_thresholds = self.model_calibrator.calibrate_impact_thresholds(
                self.pipeline, X, y
            )
            self.impact_thresholds = calibrated_thresholds
            
            # Atualizar as métricas com os resultados após calibração
            if "metrics" in result:
                logger.info("Atualizando métricas com resultados pós-calibração...")
                updated_metrics = self.evaluate(X, y)
                result["metrics"] = updated_metrics
                result["calibrated"] = True
                
                # Registrar melhoria após calibração
                if "accuracy" in updated_metrics and "accuracy" in result["metrics"]:
                    improvement = updated_metrics["accuracy"] - result["metrics"]["accuracy"]
                    logger.info(f"Melhoria de acurácia após calibração: {improvement:.4f}")
                    result["calibration_improvement"] = improvement
        
        return result
        
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza predição de impacto para um único exemplo.
        
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
            
            # Obter probabilidades
            if self.calibrated_model and self.calibrate_probabilities:
                probabilities = self.calibrated_model.predict_proba(features)[0]
            else:
                probabilities = self.model.predict_proba(features)[0]
                
            # Criar mapeamento de classes para probabilidades
            class_probs = {str(cls): float(prob) for cls, prob in zip(self.classes_, probabilities)}
            
            # Converter código da classe para nome legível
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
                      
            # Identificar as características mais importantes para a predição
            if hasattr(self.model, 'steps'):
                # Obter o classificador final do pipeline
                classifier = self.model.steps[-1][1]
                if hasattr(classifier, 'feature_importances_'):
                    feature_importances = classifier.feature_importances_
                else:
                    feature_importances = None
            else:
                feature_importances = self.model.feature_importances_ if hasattr(self.model, 'feature_importances_') else None
                    
            top_features = []
            if feature_importances is not None:
                # Obter índices das características mais importantes
                if self.select_features and hasattr(self.feature_selector, 'get_support'):
                    # Considerar apenas características selecionadas
                    selected_indices = self.feature_selector.get_support(indices=True)
                    top_indices = selected_indices[np.argsort(feature_importances)[-5:]]
                else:
                    # Usar todas as características
                    top_indices = np.argsort(feature_importances)[-5:]
                    
                if hasattr(self, 'feature_names_') and self.feature_names_ is not None:
                    # Mapear índices para nomes de características
                    top_features = [
                        (self.feature_names_[i], float(feature_importances[i if self.select_features else i]))
                        for i in top_indices
                    ]
                else:
                    top_features = [
                        (f"feature_{i}", float(feature_importances[i if self.select_features else i]))
                        for i in top_indices
                    ]
                    
                # Ordenar por importância
                top_features = sorted(top_features, key=lambda x: x[1], reverse=True)
                
            # Resultado da predição
            result = {
                "predicted_class": predicted_class_str,
                "readable_class": readable_class,
                "probabilities": class_probs,
                "confidence": float(max(probabilities)),
                "inference_time": inference_time,
                "important_features": top_features
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}", exc_info=True)
            return {
                "error": f"Erro na predição: {str(e)}",
                "error_code": "PREDICTION_ERROR",
                "traceback": str(e)
            }
    
    def get_impact_explanation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera uma explicação detalhada da predição de impacto.
        
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
        if "normalized_text" in data:
            text = data["normalized_text"]
        elif "original_text" in data:
            text = data["original_text"]
        else:
            return {"error": "Texto não encontrado para explicação"}
            
        # Extrair características para análise
        features = {}
        try:
            # Extrair características com o extrator composto
            features = self.feature_extractor.extract(text, tokens=data.get("tokens", {}))
            
            # Calcular características compostas
            composite_features = self.composite_calculator.calculate(features)
            features.update(composite_features)
            
        except Exception as e:
            logger.error(f"Erro ao extrair características para explicação: {str(e)}")
            features = {}
            
        # Preparar explicação
        impact_class = prediction["readable_class"]
        confidence = prediction["confidence"]
        explanation = {
            "impact": impact_class,
            "confidence": confidence,
            "summary": self.impact_explainer.generate_summary(prediction, features),
            "contributing_factors": []
        }
        
        # Extrair fatores que contribuem para a predição
        try:
            # 1. Valores percentuais
            if features.get("percentage_count", 0) > 0:
                percentage_info = {
                    "type": "percentages",
                    "description": "Valores percentuais identificados",
                    "details": {
                        "count": features.get("percentage_count", 0),
                        "max": features.get("max_percentage", 0),
                        "avg": features.get("avg_percentage", 0),
                        "high_count": features.get("high_percentage_count", 0),
                        "medium_count": features.get("medium_percentage_count", 0),
                        "low_count": features.get("low_percentage_count", 0)
                    }
                }
                explanation["contributing_factors"].append(percentage_info)
                
            # 2. Termos de impacto
            impact_terms_found = False
            for category in ["high_impact", "medium_impact", "low_impact"]:
                if features.get(f"{category}_count", 0) > 0:
                    impact_terms_found = True
                    break
                    
            if impact_terms_found:
                impact_terms_info = {
                    "type": "impact_terms",
                    "description": "Termos indicativos de impacto",
                    "details": {
                        "high_impact_count": features.get("high_impact_count", 0),
                        "medium_impact_count": features.get("medium_impact_count", 0),
                        "low_impact_count": features.get("low_impact_count", 0),
                        "high_to_low_ratio": features.get("high_to_low_ratio", 0)
                    }
                }
                explanation["contributing_factors"].append(impact_terms_info)
                
            # 3. Orientação temporal
            timeframe = features.get("predominant_timeframe", "none")
            if timeframe != "none":
                temporal_info = {
                    "type": "temporal",
                    "description": "Orientação temporal predominante",
                    "details": {
                        "timeframe": timeframe,
                        "short_term_count": features.get("short_term_count", 0),
                        "medium_term_count": features.get("medium_term_count", 0),
                        "long_term_count": features.get("long_term_count", 0)
                    }
                }
                explanation["contributing_factors"].append(temporal_info)
                
            # 4. Entidades financeiras
            entity_count = features.get("total_entity_count", 0)
            if entity_count > 0:
                entities_info = {
                    "type": "entities",
                    "description": "Entidades financeiras identificadas",
                    "details": {
                        "total_count": entity_count,
                        "money_count": features.get("entity_MONEY_count", 0),
                        "percent_count": features.get("entity_PERCENT_count", 0),
                        "organization_count": features.get("entity_ORG_count", 0)
                    }
                }
                explanation["contributing_factors"].append(entities_info)
                
            # 5. Pontuação composta
            composite_score = features.get("composite_impact_score", 0)
            if composite_score > 0:
                composite_info = {
                    "type": "composite_score",
                    "description": "Análise composta de impacto",
                    "details": {
                        "overall_score": composite_score,
                        "percentage_component": features.get("percentage_impact_score", 0),
                        "terms_component": features.get("terms_impact_score", 0),
                        "temporal_component": features.get("temporal_impact_score", 0),
                        "volatility_index": features.get("volatility_index", 0)
                    }
                }
                explanation["contributing_factors"].append(composite_info)
                
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
                        "confidence_level": self.impact_explainer.interpret_confidence_margin(margin)
                    }
                    
            # Incluir características importantes do modelo
            if "important_features" in prediction and prediction["important_features"]:
                feature_info = {
                    "type": "model_features",
                    "description": "Características importantes para o modelo",
                    "features": prediction["important_features"]
                }
                explanation["contributing_factors"].append(feature_info)
                
        except Exception as e:
            logger.error(f"Erro ao gerar explicação detalhada: {str(e)}")
            explanation["error_in_explanation"] = str(e)
            
        return explanation
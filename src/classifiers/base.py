"""
Módulo base para classificadores do sistema FinSemble.

Este módulo define a classe base abstrata para todos os classificadores
especializados do sistema, estabelecendo a interface comum e funcionalidades
compartilhadas para treinamento, predição e avaliação.
"""

import os
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

# Configuração de logging
logger = logging.getLogger(__name__)

class BaseClassifier(ABC):
    """
    Classe base abstrata para todos os classificadores do FinSemble.
    
    Esta classe define a interface comum e implementa funcionalidades
    compartilhadas para todos os classificadores especializados.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o classificador com as configurações fornecidas.
        
        Args:
            config: Dicionário com configurações para o classificador
        """
        self.config = config
        self.name = config.get("name", self.__class__.__name__)
        self.model_path = config.get("model_path", "models")
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.calibrate_probabilities = config.get("calibrate_probabilities", True)
        
        # Atributos inicializados durante o treinamento
        self.model = None
        self.calibrated_model = None
        self.classes_ = None
        self.feature_names_ = None
        self.is_trained = False
        
        # Estatísticas e metadados
        self.stats = {
            "training_count": 0,
            "inference_count": 0,
            "total_training_time": 0,
            "total_inference_time": 0,
            "avg_training_time": 0,
            "avg_inference_time": 0
        }
        
        self.training_metadata = {}
        logger.info(f"Classificador {self.name} inicializado com configuração: {config}")
    
    @abstractmethod
    def _create_model(self):
        """
        Cria e configura o modelo de aprendizado de máquina.
        
        Returns:
            O modelo configurado
        """
        pass
    
    @abstractmethod
    def _validate_input(self, preprocessed_data: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Valida os dados de entrada para processamento.
        
        Args:
            preprocessed_data: Dados preprocessados para validação
            
        Returns:
            Tupla com (é_válido, mensagem_erro)
        """
        pass
    
    @abstractmethod
    def _extract_features(self, preprocessed_data: Dict[str, Any]) -> np.ndarray:
        """
        Extrai características dos dados preprocessados.
        
        Args:
            preprocessed_data: Resultado do preprocessador universal
            
        Returns:
            Array numpy com características extraídas
        """
        pass
    
    def _prepare_training_features(self, training_data: List[Dict[str, Any]]) -> np.ndarray:
        """
        Prepara características para treinamento a partir dos dados preprocessados.
        
        Args:
            training_data: Lista de dados preprocessados
            
        Returns:
            Array numpy com características para treinamento
        """
        features = []
        for data in training_data:
            feature_vector = self._extract_features(data)
            features.append(feature_vector)
        return np.array(features)
    
    def train(self, training_data: List[Dict[str, Any]], labels: List[Any], 
             validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Treina o classificador com os dados fornecidos.
        
        Args:
            training_data: Lista de dados preprocessados para treinamento
            labels: Lista de rótulos correspondentes
            validation_split: Proporção de dados para validação
            
        Returns:
            Resultados do treinamento
        """
        start_time = time.time()
        logger.info(f"Iniciando treinamento do classificador {self.name}")
        
        try:
            # Criar e configurar o modelo
            self.model = self._create_model()
            
            # Extrair características
            X = self._prepare_training_features(training_data)
            y = np.array(labels)
            
            # Registrar classes únicas
            self.classes_ = np.unique(y)
            
            # Dividir dados para validação
            if validation_split > 0:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=validation_split, stratify=y, random_state=42
                )
            else:
                X_train, y_train = X, y
                X_val, y_val = None, None
                
            # Treinar o modelo
            self.model.fit(X_train, y_train)
            
            # Calibrar probabilidades, se configurado
            if self.calibrate_probabilities and X_val is not None and len(X_val) > 0:
                self.calibrated_model = CalibratedClassifierCV(cv="prefit")
                self.calibrated_model.fit(X_val, y_val)
                
            # Registrar que o modelo está treinado
            self.is_trained = True
            self.stats["training_count"] += 1
            
            # Calcular métricas, se houver dados de validação
            metrics = {}
            if X_val is not None and len(X_val) > 0:
                predictions = self.model.predict(X_val)
                metrics = {
                    "accuracy": accuracy_score(y_val, predictions),
                    "precision": precision_score(y_val, predictions, average='weighted'),
                    "recall": recall_score(y_val, predictions, average='weighted'),
                    "f1": f1_score(y_val, predictions, average='weighted')
                }
                
                logger.info(f"Métricas de validação: Acurácia={metrics['accuracy']:.4f}, "
                          f"F1={metrics['f1']:.4f}")
            
            # Atualizar estatísticas
            training_time = time.time() - start_time
            self.stats["total_training_time"] += training_time
            self.stats["avg_training_time"] = self.stats["total_training_time"] / self.stats["training_count"]
            
            # Registrar metadados de treinamento
            self.training_metadata = {
                "num_samples": len(y),
                "num_features": X.shape[1] if hasattr(X, "shape") else None,
                "class_distribution": {str(cls): int(np.sum(y == cls)) for cls in self.classes_},
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "training_time": training_time
            }
            
            logger.info(f"Treinamento do classificador {self.name} concluído em {training_time:.2f} segundos")
            
            # Retornar informações do treinamento
            return {
                "success": True,
                "training_time": training_time,
                "num_samples": len(y),
                "num_features": X.shape[1] if hasattr(X, "shape") else None,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Erro no treinamento do classificador {self.name}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_code": "TRAINING_ERROR"
            }
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Realiza predição em novos dados.
        
        Args:
            data: Dados preprocessados para classificação
            
        Returns:
            Dicionário com a classe prevista e probabilidades
        """
        # Verificar se o modelo está treinado
        if not self.is_trained:
            logger.warning(f"Classificador {self.name} não treinado")
            return {"error": "Modelo não treinado", "error_code": "MODEL_NOT_TRAINED"}
            
        # Validar entrada
        is_valid, error = self._validate_input(data)
        if not is_valid:
            return error
            
        start_time = time.time()
        logger.debug(f"Iniciando predição com {self.name}")
        
        try:
            # Extrair características
            features = self._extract_features(data)
            
            # Verificar dimensão das características
            if len(features.shape) < 2:
                # Reshapear para batch de uma amostra
                features = features.reshape(1, -1)
            
            # Realizar predição
            predicted_class = self.model.predict(features)[0]
            
            # Obter probabilidades
            if self.calibrated_model and self.calibrate_probabilities:
                probabilities = self.calibrated_model.predict_proba(features)[0]
            else:
                probabilities = self.model.predict_proba(features)[0]
                
            # Criar mapeamento de classes para probabilidades
            class_probs = {str(cls): float(prob) for cls, prob in zip(self.classes_, probabilities)}
            
            # Atualizar estatísticas
            inference_time = time.time() - start_time
            self.stats["inference_count"] += 1
            self.stats["total_inference_time"] += inference_time
            self.stats["avg_inference_time"] = (
                self.stats["total_inference_time"] / self.stats["inference_count"]
            )
            
            logger.debug(f"Predição concluída em {inference_time*1000:.2f}ms. "
                       f"Classe: {predicted_class}, Confiança: {max(probabilities):.4f}")
            
            # Resultado da predição
            return {
                "predicted_class": str(predicted_class),
                "probabilities": class_probs,
                "confidence": float(max(probabilities)),
                "inference_time": inference_time
            }
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}", exc_info=True)
            return {
                "error": f"Erro na predição: {str(e)}",
                "error_code": "PREDICTION_ERROR"
            }
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Avalia o desempenho do modelo em um conjunto de dados.
        
        Args:
            X: Características para avaliação
            y: Rótulos verdadeiros
            
        Returns:
            Dicionário com métricas de desempenho
        """
        if not self.is_trained:
            logger.warning(f"Classificador {self.name} não treinado")
            return {"error": "Modelo não treinado", "error_code": "MODEL_NOT_TRAINED"}
            
        try:
            # Fazer predições
            predictions = self.model.predict(X)
            
            # Calcular métricas
            metrics = {
                "accuracy": accuracy_score(y, predictions),
                "precision": precision_score(y, predictions, average='weighted'),
                "recall": recall_score(y, predictions, average='weighted'),
                "f1": f1_score(y, predictions, average='weighted')
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Erro na avaliação: {str(e)}", exc_info=True)
            return {
                "error": f"Erro na avaliação: {str(e)}",
                "error_code": "EVALUATION_ERROR"
            }
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Salva o modelo treinado em disco.
        
        Args:
            filepath: Caminho para salvar o modelo (opcional)
            
        Returns:
            Caminho onde o modelo foi salvo
        """
        import pickle
        
        if not self.is_trained:
            logger.warning(f"Tentativa de salvar classificador {self.name} não treinado")
            raise ValueError("Modelo não treinado")
            
        if filepath is None:
            os.makedirs(self.model_path, exist_ok=True)
            filepath = os.path.join(self.model_path, f"{self.name}.pkl")
            
        # Preparar objeto para serialização
        model_data = {
            "model": self.model,
            "calibrated_model": self.calibrated_model,
            "classes_": self.classes_,
            "feature_names_": self.feature_names_,
            "is_trained": self.is_trained,
            "training_metadata": self.training_metadata,
            "stats": self.stats
        }
        
        # Salvar arquivo
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"Modelo salvo em {filepath}")
        return filepath
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Carrega um modelo treinado do disco.
        
        Args:
            filepath: Caminho para carregar o modelo (opcional)
            
        Returns:
            True se o carregamento for bem-sucedido, False caso contrário
        """
        import pickle
        
        if filepath is None:
            filepath = os.path.join(self.model_path, f"{self.name}.pkl")
            
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # Restaurar estado do modelo
            self.model = model_data["model"]
            self.calibrated_model = model_data.get("calibrated_model")
            self.classes_ = model_data.get("classes_")
            self.feature_names_ = model_data.get("feature_names_")
            self.is_trained = model_data.get("is_trained", True)
            self.training_metadata = model_data.get("training_metadata", {})
            self.stats = model_data.get("stats", self.stats)
            
            logger.info(f"Modelo carregado de {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de {filepath}: {str(e)}")
            return False
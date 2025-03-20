"""
Módulo para calibração e otimização de modelos de machine learning.

Este módulo contém ferramentas para calibração de parâmetros,
seleção de características e otimização de modelos.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Configuração de logging
logger = logging.getLogger(__name__)

class ModelCalibrator:
    """Calibrador de parâmetros e limiares para modelos de classificação."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o calibrador de modelos.
        
        Args:
            config: Configurações para o calibrador
        """
        self.config = config or {}
        self.calibration_folds = config.get("calibration_folds", 5)
        self.n_jobs = config.get("n_jobs", -1)
    
    def calibrate_model_parameters(self, pipeline, X: np.ndarray, y: np.ndarray, 
                                 param_grid: Optional[Dict[str, List[Any]]] = None) -> Dict[str, Any]:
        """
        Calibra parâmetros do modelo usando validação cruzada.
        
        Args:
            pipeline: Pipeline do sklearn a ser calibrado
            X: Matriz de características de treinamento
            y: Rótulos de classe
            param_grid: Grade de parâmetros para otimização
            
        Returns:
            Dicionário com resultados da calibração
        """
        logger.info("Iniciando calibração de parâmetros...")
        
        # Definir grade de parâmetros padrão se não fornecida
        if param_grid is None:
            param_grid = {
                'classifier__var_smoothing': [1e-11, 1e-10, 1e-9, 1e-8, 1e-7]
            }
            
        # Criar validação cruzada estratificada
        cv = StratifiedKFold(n_splits=self.calibration_folds, shuffle=True, random_state=42)
        
        # Criar grid search com pipeline
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=self.n_jobs if self.n_jobs > 0 else None
        )
        
        # Executar busca em grade
        grid_search.fit(X, y)
        
        # Registrar resultados
        logger.info(f"Melhores parâmetros encontrados: {grid_search.best_params_}")
        logger.info(f"Melhor pontuação: {grid_search.best_score_:.4f}")
        
        # Retornar resultados
        return {
            "best_estimator": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
    
    def calibrate_impact_thresholds(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Calibra os limiares de impacto com base nas probabilidades.
        
        Args:
            model: Modelo treinado
            X: Matriz de características
            y: Rótulos verdadeiros
            
        Returns:
            Dicionário com limiares calibrados por classe
        """
        logger.info("Calibrando limiares de impacto...")
        
        # Obter probabilidades para todas as amostras
        probas = model.predict_proba(X)
        
        # Mapear índices de classes para strings
        class_indices = {cls: i for i, cls in enumerate(model.classes_)}
        
        # Limiares calibrados
        calibrated_thresholds = {}
        
        # Para cada classe, encontrar o limiar que otimiza a precisão
        for cls, idx in class_indices.items():
            # Extrair probabilidades para esta classe
            cls_probas = probas[:, idx]
            
            # Criar array de rótulos binários (1 para esta classe, 0 para outras)
            binary_y = np.array([1 if label == cls else 0 for label in y])
            
            # Testar diferentes limiares
            best_threshold = 0.5  # valor padrão
            best_f1 = 0
            
            for threshold in np.arange(0.3, 0.9, 0.05):
                # Criar predições binárias com este limiar
                preds = (cls_probas >= threshold).astype(int)
                
                # Calcular precisão, recall e F1
                true_positives = np.sum((preds == 1) & (binary_y == 1))
                false_positives = np.sum((preds == 1) & (binary_y == 0))
                false_negatives = np.sum((preds == 0) & (binary_y == 1))
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    
            # Armazenar limiar para esta classe
            calibrated_thresholds[cls] = best_threshold
            logger.info(f"Limiar calibrado para classe '{cls}': {best_threshold:.2f} (F1: {best_f1:.4f})")
            
        return calibrated_thresholds


class EnhancedGaussianNB:
    """
    Implementação aprimorada do Gaussian Naive Bayes com normalização automática
    e cálculo de importância de características.
    """
    
    def __init__(self, var_smoothing=1e-9, normalize=True, priors=None):
        """
        Inicializa o classificador Gaussian NB aprimorado.
        
        Args:
            var_smoothing: Porção da maior variância para adicionar a variâncias para estabilidade
            normalize: Se deve aplicar normalização às características
            priors: Probabilidades a priori para as classes
        """
        from sklearn.naive_bayes import GaussianNB
        from sklearn.preprocessing import StandardScaler
        
        self.var_smoothing = var_smoothing
        self.normalize = normalize
        self.priors = priors
        self.model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        self.scaler = StandardScaler() if normalize else None
        self.feature_importances_ = None
        self.classes_ = None
    
    def fit(self, X, y):
        """
        Treina o classificador com normalização opcional.
        
        Args:
            X: Array de características de treinamento
            y: Rótulos de classe
            
        Returns:
            self: O classificador treinado
        """
        # Aplicar normalização, se configurada
        if self.normalize:
            X = self.scaler.fit_transform(X)
        
        # Treinar o classificador
        self.model.fit(X, y)
        
        # Armazenar classes
        self.classes_ = self.model.classes_
        
        # Calcular importância das características após o treinamento
        self._calculate_feature_importance()
        
        return self
    
    def predict(self, X):
        """
        Realiza predição com normalização opcional.
        
        Args:
            X: Array de características para predição
            
        Returns:
            Array com classes previstas
        """
        if self.normalize:
            X = self.scaler.transform(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Calcula probabilidades de classe com normalização opcional.
        
        Args:
            X: Array de características para predição
            
        Returns:
            Array com probabilidades por classe
        """
        if self.normalize:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)
    
    def _calculate_feature_importance(self):
        """
        Calcula a importância das características com base nas diferenças de médias
        entre classes, ponderadas pelo inverso da variância.
        """
        if not hasattr(self.model, 'theta_') or not hasattr(self.model, 'var_'):
            logger.warning("Modelo não está treinado ou não possui atributos necessários para calcular importância")
            self.feature_importances_ = None
            return
        
        n_classes = len(self.classes_)
        n_features = self.model.theta_.shape[1]
        
        # Calcular a diferença média ponderada entre classes
        importance = np.zeros(n_features)
        
        # Para cada par de classes
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                # Calcular a diferença absoluta entre médias
                mean_diff = np.abs(self.model.theta_[i] - self.model.theta_[j])
                
                # Calcular a média das variâncias para cada característica
                avg_var = (self.model.var_[i] + self.model.var_[j]) / 2.0
                
                # Calcular a importância como a diferença de médias normalizada pela variância
                # Usar epsilon para evitar divisão por zero
                epsilon = 1e-10
                feature_imp = mean_diff / (np.sqrt(avg_var) + epsilon)
                
                # Acumular importância
                importance += feature_imp
        
        # Normalizar para que a soma seja 1
        importance_sum = np.sum(importance)
        if importance_sum > 0:
            self.feature_importances_ = importance / importance_sum
        else:
            self.feature_importances_ = np.ones(n_features) / n_features
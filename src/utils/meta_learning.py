"""
Módulo para técnicas de meta-aprendizado no sistema FinSemble.

Este módulo contém implementações de técnicas de meta-aprendizado
para otimizar a agregação de classificadores especializados.
"""
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Configuração de logging
logger = logging.getLogger(__name__)

class ModelConfidenceEstimator:
    """Estima a confiança de cada modelo em diferentes contextos."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o estimador de confiança.
        
        Args:
            config: Configurações para o estimador
        """
        self.config = config or {}
        self.confidence_matrix = {}
        self.history_size = config.get("history_size", 100)
        self.history = []
        self.model_weights = {}
        self.default_weight = 1.0
        
    def update(self, results: Dict[str, Any], ground_truth: Optional[Dict[str, Any]] = None):
        """
        Atualiza as estimativas de confiança com base em novos resultados.
        
        Args:
            results: Resultados dos classificadores
            ground_truth: Valores verdadeiros, se disponíveis
        """
        if ground_truth:
            # Se temos valores verdadeiros, atualizar com base na acurácia
            self._update_with_ground_truth(results, ground_truth)
        else:
            # Caso contrário, usar heurísticas como nível de confiança relatado
            self._update_with_heuristics(results)
            
        # Atualizar história
        self.history.append(results)
        if len(self.history) > self.history_size:
            self.history.pop(0)
            
        # Recalcular pesos dos modelos
        self._recalculate_weights()
        
    def _update_with_ground_truth(self, results: Dict[str, Any], ground_truth: Dict[str, Any]):
        """
        Atualiza confiança usando valores verdadeiros.
        
        Args:
            results: Resultados dos classificadores
            ground_truth: Valores verdadeiros
        """
        for model_name, model_result in results.items():
            if model_name not in self.confidence_matrix:
                self.confidence_matrix[model_name] = {}
                
            predicted_class = model_result.get("predicted_class")
            true_class = ground_truth.get(model_name)
            
            if predicted_class and true_class:
                # Calcular acurácia
                is_correct = predicted_class == true_class
                
                # Atualizar confiança para esta classe
                if predicted_class not in self.confidence_matrix[model_name]:
                    self.confidence_matrix[model_name][predicted_class] = {"correct": 0, "total": 0}
                    
                self.confidence_matrix[model_name][predicted_class]["total"] += 1
                if is_correct:
                    self.confidence_matrix[model_name][predicted_class]["correct"] += 1
                    
    def _update_with_heuristics(self, results: Dict[str, Any]):
        """
        Atualiza confiança usando heurísticas como confiança relatada.
        
        Args:
            results: Resultados dos classificadores
        """
        for model_name, model_result in results.items():
            if model_name not in self.confidence_matrix:
                self.confidence_matrix[model_name] = {}
                
            predicted_class = model_result.get("predicted_class")
            confidence = model_result.get("confidence", 0.5)
            
            if predicted_class:
                # Usar confiança relatada para ajustar
                if predicted_class not in self.confidence_matrix[model_name]:
                    self.confidence_matrix[model_name][predicted_class] = {"confidence": 0.0, "samples": 0}
                    
                # Atualizar média móvel da confiança
                current = self.confidence_matrix[model_name][predicted_class]
                n = current["samples"]
                if n > 0:
                    current["confidence"] = (current["confidence"] * n + confidence) / (n + 1)
                else:
                    current["confidence"] = confidence
                    
                current["samples"] += 1
       
    def update_with_feedback(self, model_name: str, class_name: str, was_correct: bool):
        """
        Atualiza as estimativas com feedback explícito sobre uma predição.
        
        Args:
            model_name: Nome do modelo
            class_name: Classe predita
            was_correct: Indicador se a predição estava correta
        """
        if model_name not in self.confidence_matrix:
            self.confidence_matrix[model_name] = {}
            
        if class_name not in self.confidence_matrix[model_name]:
            self.confidence_matrix[model_name][class_name] = {"correct": 0, "total": 0}
            
        self.confidence_matrix[model_name][class_name]["total"] += 1
        if was_correct:
            self.confidence_matrix[model_name][class_name]["correct"] += 1
            
        # Recalcular pesos
        self._recalculate_weights()
                 
    def _recalculate_weights(self):
        """
        Recalcula os pesos dos modelos com base nas evidências históricas.
        Usa uma abordagem bayesiana para atualizar as estimativas de confiança.
        """
        for model_name, class_confidences in self.confidence_matrix.items():
            # Inicializar com priori beta (1, 1) - equivalente a distribuição uniforme
            alpha, beta = 1, 1
            
            for class_name, data in class_confidences.items():
                if "correct" in data and data["total"] > 0:
                    # Abordagem bayesiana: atualizar distribuição beta
                    # Alpha = sucessos + 1, Beta = falhas + 1
                    alpha += data["correct"]
                    beta += data["total"] - data["correct"]
                elif "confidence" in data and data["samples"] > 0:
                    # Converter confiança relatada para parâmetros beta
                    conf = data["confidence"]
                    # Usar confiança como média da distribuição beta
                    # E número de amostras para definir "força" da crença
                    strength = min(data["samples"], 10)  # Limitar influência
                    alpha += conf * strength
                    beta += (1 - conf) * strength
            
            # Calcular média da distribuição beta: alpha / (alpha + beta)
            # Esta é a estimativa do peso do modelo
            self.model_weights[model_name] = alpha / (alpha + beta)
                
    def get_model_weight(self, model_name: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Obtém o peso de um modelo em um determinado contexto.
        
        Args:
            model_name: Nome do modelo
            context: Contexto para a predição (opcional)
            
        Returns:
            Peso do modelo (0-1)
        """
        # Se temos informações específicas de contexto, usá-las
        if context and model_name in self.confidence_matrix:
            # Identificar aspectos relevantes do contexto
            category = context.get("category")
            
            if category and category in self.confidence_matrix[model_name]:
                if "correct" in self.confidence_matrix[model_name][category]:
                    data = self.confidence_matrix[model_name][category]
                    if data["total"] > 0:
                        return data["correct"] / data["total"]
                elif "confidence" in self.confidence_matrix[model_name][category]:
                    return self.confidence_matrix[model_name][category]["confidence"]
                    
        # Caso contrário, usar peso geral do modelo
        return self.model_weights.get(model_name, self.default_weight)
        
    def get_all_weights(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Obtém os pesos de todos os modelos.
        
        Args:
            context: Contexto para a predição (opcional)
            
        Returns:
            Dicionário com pesos por modelo
        """
        if context:
            return {model: self.get_model_weight(model, context) 
                  for model in self.confidence_matrix}
        else:
            return self.model_weights.copy()


class MetaClassifier:
    """
    Implementa técnicas de meta-aprendizado para otimizar a agregação.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o meta-classificador.
        
        Args:
            config: Configurações para o meta-classificador
        """
        self.config = config or {}
        self.method = config.get("method", "weighted_average")
        self.confidence_estimator = ModelConfidenceEstimator(config)
        self.learning_rate = config.get("learning_rate", 0.01)
        
    def combine_predictions(self, model_outputs: Dict[str, Dict[str, Any]], 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combina predições de múltiplos modelos usando meta-aprendizado.
        
        Args:
            model_outputs: Saídas dos classificadores {model_name: {outputs}}
            context: Informações de contexto para a predição
            
        Returns:
            Predição combinada
        """
        if self.method == "weighted_average":
            return self._weighted_average(model_outputs, context)
        elif self.method == "stacking":
            return self._stacking(model_outputs, context)
        else:
            logger.warning(f"Método '{self.method}' não implementado, usando weighted_average")
            return self._weighted_average(model_outputs, context)
            
    def _weighted_average(self, model_outputs: Dict[str, Dict[str, Any]], 
                        context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combina predições usando média ponderada.
        
        Args:
            model_outputs: Saídas dos classificadores
            context: Informações de contexto
            
        Returns:
            Predição combinada
        """
        # Obter pesos dos modelos
        weights = self.confidence_estimator.get_all_weights(context)
        
        # Combinar probabilidades por classe
        combined_probas = {}
        normalizer = {}
        
        for model_name, output in model_outputs.items():
            if "probabilities" not in output:
                logger.warning(f"Modelo '{model_name}' não forneceu probabilidades, ignorando")
                continue
                
            model_weight = weights.get(model_name, 1.0)
            
            # Combinar probabilidades de cada classe
            for class_name, prob in output["probabilities"].items():
                if class_name not in combined_probas:
                    combined_probas[class_name] = 0.0
                    normalizer[class_name] = 0.0
                    
                combined_probas[class_name] += prob * model_weight
                normalizer[class_name] += model_weight
                
        # Normalizar probabilidades
        for class_name in combined_probas:
            if normalizer[class_name] > 0:
                combined_probas[class_name] /= normalizer[class_name]
                
        # Encontrar classe com maior probabilidade
        if combined_probas:
            predicted_class = max(combined_probas.items(), key=lambda x: x[1])[0]
            confidence = combined_probas[predicted_class]
        else:
            # Fallback se não foi possível combinar probabilidades
            predicted_class = "unknown"
            confidence = 0.0
            
        # Construir resultado
        result = {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": combined_probas,
            "method": "weighted_average",
            "model_weights": weights
        }
        
        return result
        
    def _stacking(self, model_outputs: Dict[str, Dict[str, Any]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combina predições usando stacking (meta-modelo).
        
        Args:
            model_outputs: Saídas dos classificadores
            context: Informações de contexto
            
        Returns:
            Predição combinada
        """
        # Nota: Esta é uma implementação simplificada de stacking
        # Em uma implementação real, treianaríamos um meta-modelo em dados históricos
        
        # Por enquanto, usaremos uma abordagem baseada em regras simples
        # Classificador de tipo tem maior peso para determinar tipo
        # Classificador de sentimento tem maior peso para sentimento
        # Classificador de impacto tem maior peso para impacto
        
        # Extrair predições por dimensão
        type_prediction = model_outputs.get("type_classifier", {}).get("predicted_class", "unknown")
        sentiment_prediction = model_outputs.get("sentiment_analyzer", {}).get("predicted_class", "neutral")
        impact_prediction = model_outputs.get("impact_modeler", {}).get("predicted_class", "medium")
        
        # Construir resultado combinado
        combined_result = {
            "predicted_classes": {
                "type": type_prediction,
                "sentiment": sentiment_prediction,
                "impact": impact_prediction
            },
            "confidences": {
                "type": model_outputs.get("type_classifier", {}).get("confidence", 0.0),
                "sentiment": model_outputs.get("sentiment_analyzer", {}).get("confidence", 0.0),
                "impact": model_outputs.get("impact_modeler", {}).get("confidence", 0.0)
            },
            "method": "stacking"
        }
        
        # Calcular confiança média
        confidences = [conf for conf in combined_result["confidences"].values() if conf > 0]
        combined_result["overall_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        return combined_result
        
    def update(self, model_outputs: Dict[str, Dict[str, Any]], 
              ground_truth: Optional[Dict[str, Any]] = None):
        """
        Atualiza o meta-classificador com novos dados.
        
        Args:
            model_outputs: Saídas dos classificadores
            ground_truth: Valores verdadeiros, se disponíveis
        """
        # Atualizar estimador de confiança
        self.confidence_estimator.update(model_outputs, ground_truth)
        
        # Se tivéssemos um meta-modelo treinável, atualizaríamos aqui
        
class AdvancedMetaClassifier(MetaClassifier):
    """
    Implementação avançada de meta-classificação com técnicas de stacking
    usando modelos treinados com dados históricos.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o meta-classificador avançado.
        
        Args:
            config: Configurações para o meta-classificador
        """
        super().__init__(config)
        self.stacking_model = None
        self.feature_names = []
        self.historical_data = []
        self.max_history_size = config.get("max_history_size", 1000)
        
    def _extract_features_from_outputs(self, model_outputs: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """
        Extrai características dos outputs dos modelos para usar no stacking.
        
        Args:
            model_outputs: Saídas dos classificadores
            
        Returns:
            Array com características para o meta-modelo
        """
        features = []
        self.feature_names = []
        
        # Para cada modelo, extrair características
        for model_name, output in model_outputs.items():
            # Adicionar confiança como característica
            confidence = output.get("confidence", 0.5)
            features.append(confidence)
            self.feature_names.append(f"{model_name}_confidence")
            
            # Adicionar probabilidades por classe
            if "probabilities" in output:
                for class_name, prob in output["probabilities"].items():
                    features.append(prob)
                    self.feature_names.append(f"{model_name}_{class_name}_prob")
            
            # Adicionar características específicas do modelo, se disponíveis
            if "features" in output:
                for feat_name, feat_val in output["features"].items():
                    if isinstance(feat_val, (int, float)):
                        features.append(feat_val)
                        self.feature_names.append(f"{model_name}_{feat_name}")
        
        return np.array(features)
        
    def train_stacking_model(self, historical_data: List[Dict[str, Any]]):
        """
        Treina o modelo de stacking com dados históricos.
        
        Args:
            historical_data: Lista de exemplos históricos com predições e verdade
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            
            # Extrair características e rótulos
            X = []
            y = []
            
            for example in historical_data:
                if "model_outputs" in example and "ground_truth" in example:
                    # Extrair características dos outputs dos modelos
                    X.append(self._extract_features_from_outputs(example["model_outputs"]))
                    # Extrair rótulo verdadeiro
                    y.append(example["ground_truth"])
            
            if not X or not y:
                logger.warning("Dados históricos insuficientes para treinar modelo de stacking")
                return
                
            X = np.array(X)
            y = np.array(y)
            
            # Normalizar características
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Treinar modelo de regressão logística para stacking
            model = LogisticRegression(multi_class='multinomial', max_iter=1000)
            model.fit(X_scaled, y)
            
            # Armazenar modelo e scaler
            self.stacking_model = {
                "model": model,
                "scaler": scaler,
                "classes": model.classes_
            }
            
            logger.info("Modelo de stacking treinado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao treinar modelo de stacking: {str(e)}")
            self.stacking_model = None
            
    def _stacking(self, model_outputs: Dict[str, Dict[str, Any]], 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Combina predições usando stacking com modelo treinado.
        
        Args:
            model_outputs: Saídas dos classificadores
            context: Informações de contexto
            
        Returns:
            Predição combinada
        """
        # Se temos um modelo de stacking treinado, usá-lo
        if self.stacking_model and hasattr(self.stacking_model["model"], "predict_proba"):
            try:
                # Extrair características dos outputs
                features = self._extract_features_from_outputs(model_outputs)
                
                # Normalizar
                X_scaled = self.stacking_model["scaler"].transform(features.reshape(1, -1))
                
                # Fazer predição
                probas = self.stacking_model["model"].predict_proba(X_scaled)[0]
                classes = self.stacking_model["classes"]
                
                # Selecionar classe com maior probabilidade
                predicted_class = classes[np.argmax(probas)]
                confidence = np.max(probas)
                
                # Construir resultado
                result = {
                    "predicted_class": predicted_class,
                    "confidence": float(confidence),
                    "probabilities": {str(cls): float(prob) for cls, prob in zip(classes, probas)},
                    "method": "trained_stacking",
                    "model_features": self.feature_names
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Erro ao usar modelo de stacking: {str(e)}")
                # Fallback para a implementação original
                return super()._stacking(model_outputs, context)
        else:
            # Implementação original de stacking
            return super()._stacking(model_outputs, context)
    
    def update(self, model_outputs: Dict[str, Dict[str, Any]], 
              ground_truth: Optional[Dict[str, Any]] = None):
        """
        Atualiza o meta-classificador com novos dados e treina modelo de stacking
        quando houver dados suficientes.
        
        Args:
            model_outputs: Saídas dos classificadores
            ground_truth: Valores verdadeiros, se disponíveis
        """
        # Atualizar confiança dos modelos
        super().update(model_outputs, ground_truth)
        
        # Armazenar exemplo para treinamento do modelo de stacking
        if ground_truth:
            self.historical_data.append({
                "model_outputs": model_outputs,
                "ground_truth": ground_truth.get("market_direction", "unknown")
            })
            
            # Limitar tamanho do histórico
            if len(self.historical_data) > self.max_history_size:
                self.historical_data = self.historical_data[-self.max_history_size:]
                
            # Treinar modelo de stacking quando acumular pelo menos 50 exemplos
            if len(self.historical_data) >= 50:
                self.train_stacking_model(self.historical_data)
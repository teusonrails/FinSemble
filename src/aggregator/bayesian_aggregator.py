"""
Módulo principal do Agregador Bayesiano do sistema FinSemble.

Este módulo implementa o Agregador Bayesiano, responsável por consolidar
as saídas dos classificadores especializados em uma análise multidimensional.
"""
import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import os
from concurrent.futures import ThreadPoolExecutor

from src.utils.bayesian_network import BayesianNetwork
from src.utils.meta_learning import AdvancedMetaClassifier, MetaClassifier
from src.utils.belief_propagation import BeliefPropagator, Factor

# Configuração de logging
logger = logging.getLogger(__name__)

class BayesianAggregator:
    """
    Agregador Bayesiano para o sistema FinSemble.
    
    Este componente consolida as saídas dos classificadores especializados
    em uma análise multidimensional final utilizando redes bayesianas
    e técnicas de meta-aprendizado.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o Agregador Bayesiano.
        
        Args:
            config: Configurações para o agregador, incluindo:
                - network_structure: Estrutura da rede ("auto" ou "manual")
                - inference_method: Método de inferência ("variable_elimination", "belief_propagation")
                - meta_learning: Configurações para meta-aprendizado
                - learning_rate: Taxa de aprendizado para atualização de parâmetros
                - max_iterations: Máximo de iterações para algoritmos de inferência
        """
        self.config = config
        self.name = config.get("name", "BayesianAggregator")
        self.network_structure = config.get("network_structure", "auto")
        self.inference_method = config.get("inference_method", "variable_elimination")
        self.learning_rate = config.get("learning_rate", 0.01)
        self.max_iterations = config.get("max_iterations", 100)
        
        # Configurações adicionais
        self.use_advanced_meta_learning = config.get("use_advanced_meta_learning", False)
        self.adaptive_inference = config.get("adaptive_inference", True)
        self.network_cache_enabled = config.get("network_cache_enabled", True)
        
        # Cache para resultados de inferência
        self.inference_cache = {}
        self.cache_max_size = config.get("cache_max_size", 100)
        
        # Inicializar componentes
        self._initialize_components()
        
        # Métricas e estatísticas
        self.stats = {
            "inference_count": 0,
            "total_inference_time": 0,
            "avg_inference_time": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info(f"Agregador Bayesiano inicializado com estrutura '{self.network_structure}', "
                   f"método de inferência '{self.inference_method}' e "
                   f"meta-learning avançado {'ativado' if self.use_advanced_meta_learning else 'desativado'}")
                   
    def _initialize_components(self):
        """Inicializa os componentes do Agregador Bayesiano."""
        # Criar rede bayesiana
        self.bayesian_network = BayesianNetwork("FinSembleNet")
        
        # Inicializar meta-classificador
        meta_config = self.config.get("meta_learning", {})
        if self.use_advanced_meta_learning:
            self.meta_classifier = AdvancedMetaClassifier(meta_config)
        else:
            self.meta_classifier = MetaClassifier(meta_config)
        
        # Inicializar propagador de crenças
        belief_config = {
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.config.get("convergence_threshold", 1e-6)
        }
        self.belief_propagator = BeliefPropagator(belief_config)
        
        # Inicializar estrutura da rede, se for "auto"
        if self.network_structure == "auto":
            self._initialize_auto_structure()
        else:
            logger.info("Estrutura manual da rede. Configurar manualmente.")
            
    def _initialize_auto_structure(self):
        """
        Inicializa automaticamente a estrutura da rede bayesiana.
        Versão aprimorada com CPTs mais detalhadas.
        """
        # Configuração de nós melhorada
        # Adicionar dimensões mais detalhadas para os tipos de documentos
        self.bayesian_network.add_node(
            "document_type", 
            ["report", "announcement", "guidance", "analysis", "news", "interview"]
        )
        
        # Sentimento com cinco níveis
        self.bayesian_network.add_node(
            "sentiment", 
            ["very_positive", "positive", "neutral", "negative", "very_negative"]
        )
        
        # Impacto também com mais níveis
        self.bayesian_network.add_node(
            "impact", 
            ["very_high", "high", "medium", "low", "very_low"]
        )
        
        # Nós para análise multidimensional mais detalhada
        self.bayesian_network.add_node(
            "market_direction", 
            ["strong_buy", "buy", "hold", "sell", "strong_sell"], 
            ["sentiment", "impact"]
        )
        
        self.bayesian_network.add_node(
            "investment_horizon", 
            ["immediate", "short_term", "medium_term", "long_term", "strategic"], 
            ["document_type", "impact"]
        )
        
        self.bayesian_network.add_node(
            "risk_level", 
            ["very_high", "high", "medium", "low", "very_low"], 
            ["sentiment", "impact", "document_type"]
        )
        
        self.bayesian_network.add_node(
            "confidence_level",
            ["very_high", "high", "medium", "low", "very_low"],
            ["document_type", "sentiment"]
        )
        
        # CPTs para nós sem pais (priori)
        # Distribuição mais balanceada para documento
        self.bayesian_network.set_cpt(
            "document_type", 
            np.array([0.25, 0.25, 0.15, 0.15, 0.10, 0.10])
        )
        
        # Distribuição em forma de sino para sentimento
        self.bayesian_network.set_cpt(
            "sentiment", 
            np.array([0.10, 0.20, 0.40, 0.20, 0.10])
        )
        
        # Distribuição em forma de sino para impacto
        self.bayesian_network.set_cpt(
            "impact", 
            np.array([0.10, 0.20, 0.40, 0.20, 0.10])
        )
        
        # CPTs para nós com pais
        # market_direction: matriz 5x5 (sentimento x impacto)
        # Esta é uma versão simplificada, a real seria 5x5x5
        market_direction_cpt = np.zeros((5, 5, 5))
        
        # Preencher CPT com valores mais detalhados
        # Aqui apenas um exemplo simplificado para ilustração
        for s in range(5):  # sentimento
            for i in range(5):  # impacto
                # Calcular tendência
                # s=0 (very_positive) até s=4 (very_negative)
                # i=0 (very_high) até i=4 (very_low)
                
                # Sentimento muito positivo + impacto muito alto => forte compra
                # Sentimento muito negativo + impacto muito alto => forte venda
                
                s_effect = 2 - s  # +2 para very_positive, -2 para very_negative
                i_effect = 2 - i  # +2 para very_high, -2 para very_low
                
                # Combinar efeitos
                combined = s_effect + i_effect * 0.5  # Impacto tem metade do peso do sentimento
                
                # Converter para índice (0=strong_buy a 4=strong_sell)
                direction_idx = min(max(int(2 - combined + 0.5), 0), 4)
                
                # Atribuir probabilidade alta para o índice calculado
                probs = np.ones(5) * 0.05  # Probabilidade base
                probs[direction_idx] = 0.81  # Probabilidade alta para a direção principal
                
                # Normalizar
                probs = probs / np.sum(probs)
                
                # Atribuir à CPT
                market_direction_cpt[:, s, i] = probs
        
        self.bayesian_network.set_cpt("market_direction", market_direction_cpt)
        
        # Outras CPTs seriam definidas de forma similar
        # Por brevidade, usamos distribuições uniformes para os outros nós
        
        # investment_horizon: matriz para (documento x impacto)
        horizon_shape = (5, 6, 5)  # 5 estados de horizon, 6 tipos doc, 5 níveis impacto
        investment_horizon_cpt = np.ones(horizon_shape) / horizon_shape[0]
        self.bayesian_network.set_cpt("investment_horizon", investment_horizon_cpt)
        
        # risk_level: matriz para (sentimento x impacto x documento)
        risk_shape = (5, 5, 5, 6)  # 5 níveis risco, 5 níveis sentimento, 5 níveis impacto, 6 tipos doc
        risk_level_cpt = np.ones(risk_shape) / risk_shape[0]
        self.bayesian_network.set_cpt("risk_level", risk_level_cpt)
        
        # confidence_level: matriz para (documento x sentimento)
        confidence_shape = (5, 6, 5)  # 5 níveis confiança, 6 tipos doc, 5 níveis sentimento
        confidence_level_cpt = np.ones(confidence_shape) / confidence_shape[0]
        self.bayesian_network.set_cpt("confidence_level", confidence_level_cpt)
        
        logger.info("Estrutura automática da rede bayesiana inicializada com CPTs detalhadas")
        
    def update_network_parameters(self, data: Dict[str, Any]):
        """
        Atualiza os parâmetros da rede bayesiana com novos dados.
        
        Args:
            data: Dados para atualização
        """
        # Esta é uma implementação simplificada
        # Em um sistema real, utilizaríamos algoritmos de aprendizado de parâmetros
        
        # Atualizar meta-classificador
        if "classifier_outputs" in data:
            self.meta_classifier.update(data["classifier_outputs"], data.get("ground_truth"))
            
        # Atualizar CPTs com base em observações
        # Isto seria feito com base em dados históricos e atuais
        # Por simplicidade, não implementado aqui
        logger.debug("Parâmetros da rede bayesiana atualizados")
        
    def aggregate(self, classifier_outputs: Dict[str, Dict[str, Any]], 
                 context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Agrega saídas dos classificadores em uma análise multidimensional.
        
        Args:
            classifier_outputs: Saídas dos classificadores especializados
            context: Informações de contexto (opcional)
            
        Returns:
            Análise multidimensional consolidada
        """
        start_time = time.time()
        logger.info("Iniciando agregação de resultados dos classificadores")
        
        # Validar entradas
        if not classifier_outputs:
            logger.warning("Nenhuma saída de classificador fornecida")
            return {"error": "Nenhuma saída de classificador fornecida"}
            
        try:
            ## Mapear saídas dos classificadores para o formato esperado pela rede
            mapped_outputs = self._map_classifier_outputs(classifier_outputs)
            
            # Extrair evidências dos classificadores
            evidence = self._extract_evidence(mapped_outputs)
            
            # Definir nós de consulta
            query_nodes = ["market_direction", "investment_horizon", "risk_level", "confidence_level"]
            
            # Realizar inferência na rede bayesiana
            inference_results = self._inference(evidence, query_nodes)
            
            # Aplicar meta-classificação para refinar resultados
            meta_results = self.meta_classifier.combine_predictions(classifier_outputs, context)
            
            # Combinar resultados da rede bayesiana e meta-classificação
            combined_analysis = self._combine_results(inference_results, meta_results, mapped_outputs)
            
            # Adicionar dados de explicabilidade
            combined_analysis["explanations"] = self._generate_detailed_explanations(
                classifier_outputs, inference_results, meta_results, mapped_outputs
            )
            
            # Atualizar estatísticas
            inference_time = time.time() - start_time
            self.stats["inference_count"] += 1
            self.stats["total_inference_time"] += inference_time
            self.stats["avg_inference_time"] = (
                self.stats["total_inference_time"] / self.stats["inference_count"]
            )
            
            logger.info(f"Agregação concluída em {inference_time:.3f}s")
            combined_analysis["inference_time"] = inference_time
            combined_analysis["inference_method"] = self._select_inference_method(evidence, query_nodes)
            
            return combined_analysis
            
        except Exception as e:
            logger.error(f"Erro durante agregação: {str(e)}", exc_info=True)
            return {
                "error": f"Erro durante agregação: {str(e)}",
                "inference_time": time.time() - start_time
            }
            
    def _extract_evidence(self, classifier_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Extrai evidências dos classificadores para inferência bayesiana.
        
        Args:
            classifier_outputs: Saídas dos classificadores
            
        Returns:
            Dicionário com evidências {node_name: state}
        """
        evidence = {}
        
        # Mapear saídas dos classificadores para nós da rede
        classifier_to_node = {
            "type_classifier": "document_type",
            "sentiment_analyzer": "sentiment",
            "impact_modeler": "impact"
        }
        
        # Extrair predições como evidências
        for classifier_name, node_name in classifier_to_node.items():
            if classifier_name in classifier_outputs:
                predicted_class = classifier_outputs[classifier_name].get("predicted_class")
                if predicted_class:
                    evidence[node_name] = predicted_class
                    
        logger.debug(f"Evidências extraídas: {evidence}")
        return evidence
        
    def _inference_variable_elimination(self, evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Realiza inferência usando eliminação de variáveis.
        
        Args:
            evidence: Evidências observadas
            
        Returns:
            Resultados da inferência
        """
        # Nós de consulta (os não fornecidos como evidência)
        query_nodes = [node for node in self.bayesian_network.nodes if node not in evidence]
        
        # Realizar inferência
        try:
            results = self.bayesian_network.inference(evidence, query_nodes)
            logger.debug(f"Inferência bem-sucedida para {len(query_nodes)} nós")
            return results
        except Exception as e:
            logger.error(f"Erro durante inferência por eliminação de variáveis: {str(e)}")
            # Fallback: retornar distribuições uniformes
            return {node: {state: 1.0/len(self.bayesian_network.nodes[node].states) 
                         for state in self.bayesian_network.nodes[node].states}
                  for node in query_nodes}
                  
    def _inference_belief_propagation(self, evidence: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Realiza inferência usando propagação de crenças.
        
        Args:
            evidence: Evidências observadas
            
        Returns:
            Resultados da inferência
        """
        # Esta é uma implementação simplificada
        # Na prática, converteríamos a rede para fatores e usaríamos o propagador de crenças
        
        # Por simplificidade, redirecionando para eliminação de variáveis por enquanto
        logger.info("Redirecionando para eliminação de variáveis (implementação simplificada)")
        return self._inference_variable_elimination(evidence)
        
    def _combine_results(self, inference_results: Dict[str, Dict[str, float]], 
                       meta_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combina resultados da inferência bayesiana e meta-classificação.
        
        Args:
            inference_results: Resultados da inferência bayesiana
            meta_results: Resultados da meta-classificação
            
        Returns:
            Análise combinada final
        """
        combined = {
            "bayesian_inference": inference_results,
            "meta_classification": meta_results,
            "timestamp": time.time()
        }
        
        # Extrair direção de mercado, horizonte de investimento e nível de risco
        if "market_direction" in inference_results:
            market_direction = max(inference_results["market_direction"].items(), key=lambda x: x[1])[0]
            combined["market_direction"] = market_direction
            combined["market_direction_confidence"] = inference_results["market_direction"][market_direction]
            
        if "investment_horizon" in inference_results:
            horizon = max(inference_results["investment_horizon"].items(), key=lambda x: x[1])[0]
            combined["investment_horizon"] = horizon
            combined["investment_horizon_confidence"] = inference_results["investment_horizon"][horizon]
            
        if "risk_level" in inference_results:
            risk = max(inference_results["risk_level"].items(), key=lambda x: x[1])[0]
            combined["risk_level"] = risk
            combined["risk_level_confidence"] = inference_results["risk_level"][risk]
            
        # Adicionar recomendação geral baseada na direção de mercado
        if "market_direction" in combined:
            direction = combined["market_direction"]
            if direction == "buy":
                combined["recommendation"] = "Comprar ativos de risco"
            elif direction == "sell":
                combined["recommendation"] = "Proteger capital"
            else:  # hold
                combined["recommendation"] = "Manter posições atuais"
                
        return combined
        
    def _generate_explanations(self, classifier_outputs: Dict[str, Dict[str, Any]],
                                     inference_results: Dict[str, Dict[str, float]], 
                                     meta_results: Dict[str, Any],
                                     mapped_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Gera explicações detalhadas para os resultados da agregação.
        
        Args:
            classifier_outputs: Saídas dos classificadores originais
            inference_results: Resultados da inferência bayesiana
            meta_results: Resultados da meta-classificação
            mapped_outputs: Saídas mapeadas dos classificadores
            
        Returns:
            Explicações detalhadas
        """
        explanations = {
            "summary": "",
            "factors": [],
            "classifier_confidence": {},
            "network_influences": {},
            "decision_path": []
        }
        
        # Extrair principal direção de mercado e confiança
        market_direction = "unknown"
        market_confidence = 0.0
        
        if "market_direction" in inference_results:
            market_direction = max(inference_results["market_direction"].items(), key=lambda x: x[1])[0]
            market_confidence = inference_results["market_direction"][market_direction]
            
        # Extrair horizonte de investimento
        investment_horizon = "unknown"
        if "investment_horizon" in inference_results:
            investment_horizon = max(inference_results["investment_horizon"].items(), key=lambda x: x[1])[0]
            
        # Extrair nível de risco
        risk_level = "unknown"
        if "risk_level" in inference_results:
            risk_level = max(inference_results["risk_level"].items(), key=lambda x: x[1])[0]
            
        # Gerar resumo baseado na direção, horizonte e risco
        action_mapping = {
            "strong_buy": "comprar ativos de risco agressivamente",
            "buy": "comprar ativos de risco",
            "hold": "manter posições atuais",
            "sell": "reduzir exposição a risco",
            "strong_sell": "vender ativos de risco urgentemente"
        }
        
        horizon_mapping = {
            "immediate": "imediato",
            "short_term": "curto prazo",
            "medium_term": "médio prazo",
            "long_term": "longo prazo",
            "strategic": "horizonte estratégico"
        }
        
        risk_mapping = {
            "very_high": "muito alto",
            "high": "alto",
            "medium": "médio",
            "low": "baixo",
            "very_low": "muito baixo"
        }
        
        action = action_mapping.get(market_direction, "avaliar cada posição individualmente")
        horizon = horizon_mapping.get(investment_horizon, "médio prazo")
        risk = risk_mapping.get(risk_level, "médio")
        
        confidence_level = "alta" if market_confidence > 0.8 else "moderada" if market_confidence > 0.6 else "baixa"
        
        # Gerar texto explicativo detalhado
        summary = (
            f"A análise recomenda {action} com confiança {confidence_level} ({market_confidence:.1%}). "
            f"O horizonte de investimento sugerido é de {horizon} com nível de risco {risk}."
        )
        
        explanations["summary"] = summary
        
        # Adicionar detalhes sobre o caminho de decisão
        # Mostrar como as evidências levaram à conclusão
        decision_path = []
        
        # Adicionar evidências dos classificadores
        for classifier_name, outputs in mapped_outputs.items():
            if "predicted_class" in outputs:
                decision_path.append({
                    "node": classifier_name,
                    "value": outputs["predicted_class"],
                    "confidence": outputs.get("confidence", 0.5),
                    "type": "evidence"
                })
                
        # Adicionar nós inferidos
        for node_name, distribution in inference_results.items():
            if node_name in ["market_direction", "investment_horizon", "risk_level", "confidence_level"]:
                value = max(distribution.items(), key=lambda x: x[1])[0]
                confidence = distribution[value]
                decision_path.append({
                    "node": node_name,
                    "value": value,
                    "confidence": confidence,
                    "type": "inference",
                    "distribution": {k: float(v) for k, v in sorted(distribution.items(), key=lambda x: x[1], reverse=True)[:3]}
                })
                
        explanations["decision_path"] = decision_path
        
        # Adicionar confiança dos classificadores
        for classifier_name, outputs in classifier_outputs.items():
            if "confidence" in outputs:
                explanations["classifier_confidence"][classifier_name] = outputs["confidence"]
                
                # Adicionar fatores baseados em classificadores com alta confiança
                if outputs["confidence"] > 0.7:
                    explanations["factors"].append({
                        "source": classifier_name,
                        "contribution": "strong",
                        "description": f"{classifier_name} indicou {outputs.get('predicted_class', 'desconhecido')} "
                                     f"com alta confiança ({outputs['confidence']:.1%})"
                    })
                    
        # Adicionar influências da rede bayesiana
        # Identificar nós com distribuições mais concentradas (menos entropia)
        for node_name, distribution in inference_results.items():
            # Calcular entropia normalizada (0-1, onde 0 = certeza total)
            probs = list(distribution.values())
            max_entropy = -np.log(1.0 / len(probs))  # Entropia máxima (distribuição uniforme)
            entropy = -sum(p * np.log(p) if p > 0 else 0 for p in probs)
            
            # Normalizar
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0
                
            certainty = 1.0 - normalized_entropy
            explanations["network_influences"][node_name] = certainty
            
            # Adicionar como fator se tiver alta certeza
            if certainty > 0.6:
                # Mapear nomes para português
                node_name_pt = {
                    "market_direction": "direção de mercado",
                    "investment_horizon": "horizonte de investimento",
                    "risk_level": "nível de risco",
                    "confidence_level": "nível de confiança"
                }.get(node_name, node_name)
                
                explanations["factors"].append({
                    "source": "bayesian_network",
                    "node": node_name,
                    "contribution": "informative",
                    "description": f"Análise bayesiana indica {node_name_pt} com alta certeza ({certainty:.1%})"
                })
                
        return explanations
    
    def _get_cache_key(self, evidence: Dict[str, str], query_nodes: List[str]) -> str:
        """
        Gera uma chave de cache para os resultados de inferência.
        
        Args:
            evidence: Evidências
            query_nodes: Nós de consulta
            
        Returns:
            Chave de cache
        """
        evidence_str = ",".join(f"{k}={v}" for k, v in sorted(evidence.items()))
        query_str = ",".join(sorted(query_nodes))
        return f"{evidence_str}|{query_str}|{self.inference_method}"
    
    def _select_inference_method(self, evidence: Dict[str, str], query_nodes: List[str]) -> str:
        """
        Seleciona o método de inferência mais apropriado com base no contexto.
        
        Args:
            evidence: Evidências
            query_nodes: Nós de consulta
            
        Returns:
            Método de inferência selecionado
        """
        if not self.adaptive_inference:
            return self.inference_method
            
        # Heurística para seleção de método
        network_size = len(self.bayesian_network.nodes)
        n_evidence = len(evidence)
        n_query = len(query_nodes)
        
        # Rede pequena ou consulta simples: eliminação de variáveis
        if network_size < 10 or (n_evidence + n_query > network_size * 0.7):
            return "variable_elimination"
            
        # Múltiplas consultas em rede grande: árvore de junção
        if n_query > 3 and network_size > 15:
            return "junction_tree"
            
        # Rede complexa com loops: propagação de crenças
        if any(len(self.bayesian_network.nodes[node].parents) > 2 for node in self.bayesian_network.nodes):
            return "belief_propagation"
            
        # Padrão
        return self.inference_method
 
    def _inference(self, evidence: Dict[str, str], query_nodes: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Realiza inferência na rede bayesiana usando o método mais apropriado.
        
        Args:
            evidence: Evidências observadas
            query_nodes: Nós para consulta
            
        Returns:
            Resultados da inferência
        """
        # Verificar cache
        if self.network_cache_enabled:
            cache_key = self._get_cache_key(evidence, query_nodes)
            if cache_key in self.inference_cache:
                self.stats["cache_hits"] += 1
                return self.inference_cache[cache_key]
            self.stats["cache_misses"] += 1
            
        # Selecionar método de inferência
        method = self._select_inference_method(evidence, query_nodes)
        
        # Realizar inferência com o método selecionado
        try:
            if method == "belief_propagation":
                results = self.belief_propagator.loopy_belief_propagation(
                    self.bayesian_network, evidence, query_nodes
                )
            elif method == "junction_tree":
                results = self.belief_propagator.junction_tree_inference(
                    self.bayesian_network, evidence, query_nodes
                )
            else:  # variable_elimination
                # Converter rede para fatores
                factors = []
                variable_domains = {}
                
                for node_name, node in self.bayesian_network.nodes.items():
                    variable_domains[node_name] = node.states
                    
                    # Criar fator a partir da CPT
                    factor = Factor.from_cpt(
                        node_name, 
                        node.states, 
                        node.parents, 
                        node.cpt, 
                        variable_domains
                    )
                    
                    factors.append(factor)
                    
                # Incorporar evidências
                for var, val in evidence.items():
                    # Criar fator de evidência
                    var_idx = self.bayesian_network.nodes[var].states.index(val)
                    evidence_values = np.zeros(len(self.bayesian_network.nodes[var].states))
                    evidence_values[var_idx] = 1.0
                    
                    evidence_factor = Factor([var], evidence_values, variable_domains)
                    factors.append(evidence_factor)
                    
                # Determinar ordem de eliminação ótima
                elimination_order = self._find_elimination_order(
                    self.bayesian_network, 
                    query_nodes
                )
                
                # Realizar inferência
                results = self.belief_propagator.variable_elimination(
                    factors, query_nodes, elimination_order
                )
                
            # Atualizar cache
            if self.network_cache_enabled:
                # Limitar tamanho do cache
                if len(self.inference_cache) >= self.cache_max_size:
                    # Remover item mais antigo
                    oldest_key = next(iter(self.inference_cache))
                    del self.inference_cache[oldest_key]
                    
                self.inference_cache[cache_key] = results
                
            return results
            
        except Exception as e:
            logger.error(f"Erro durante inferência: {str(e)}", exc_info=True)
            # Fallback: distribuições uniformes
            return {node: {state: 1.0 / len(self.bayesian_network.nodes[node].states) 
                         for state in self.bayesian_network.nodes[node].states}
                  for node in query_nodes}
 
    def _find_elimination_order(self, bayesian_network: BayesianNetwork, 
                              query_nodes: List[str]) -> List[str]:
        """
        Encontra uma ordem de eliminação ótima para o algoritmo de eliminação de variáveis.
        
        Args:
            bayesian_network: Rede bayesiana
            query_nodes: Nós de consulta
            
        Returns:
            Ordem de eliminação otimizada
        """
        # Construir grafo de adjacência
        graph = {node: set() for node in bayesian_network.nodes}
        
        for node_name, node in bayesian_network.nodes.items():
            # Conectar pais
            for parent in node.parents:
                graph[node_name].add(parent)
                graph[parent].add(node_name)
                
            # Conectar filhos
            for child in node.children:
                graph[node_name].add(child)
                graph[child].add(node_name)
                
        # Algoritmo de eliminação mínima: escolher variável com menor número
        # de arestas a serem adicionadas ao grafo moral
        
        # Copiar grafo e remover nós de consulta
        working_graph = {node: graph[node].copy() 
                       for node in graph if node not in query_nodes}
        
        elimination_order = []
        
        while working_graph:
            # Encontrar nó com menor número de vizinhos
            min_node = min(working_graph.items(), key=lambda x: len(x[1]))[0]
            
            # Adicionar à ordem de eliminação
            elimination_order.append(min_node)
            
            # Conectar todos os vizinhos entre si
            neighbors = list(working_graph[min_node])
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[i] in working_graph and neighbors[j] in working_graph:
                        working_graph[neighbors[i]].add(neighbors[j])
                        working_graph[neighbors[j]].add(neighbors[i])
                        
            # Remover nó do grafo
            del working_graph[min_node]
            for node in working_graph:
                if min_node in working_graph[node]:
                    working_graph[node].remove(min_node)
                    
        return elimination_order
    
    def _map_classifier_outputs(self, classifier_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Mapeia as saídas dos classificadores para o formato esperado pela rede bayesiana.
        
        Args:
            classifier_outputs: Saídas dos classificadores
            
        Returns:
            Saídas mapeadas
        """
        mapped_outputs = {}
        
        # Mapeamento para tipos de documento
        type_mapping = {
            "report": "report",
            "announcement": "announcement",
            "guidance": "guidance",
            "analysis": "analysis",
            "news": "news",
            "interview": "interview",
            # Fallbacks para valores não mapeados
            "other": "announcement",
            "unknown": "announcement"
        }
        
        # Mapeamento para sentimento
        sentiment_mapping = {
            "very_positive": "very_positive",
            "positive": "positive", 
            "neutral": "neutral",
            "negative": "negative",
            "very_negative": "very_negative",
            # Fallbacks
            "unknown": "neutral"
        }
        
        # Mapeamento para impacto
        impact_mapping = {
            "very_high": "very_high",
            "high": "high",
            "medium": "medium",
            "low": "low",
            "very_low": "very_low",
            # Fallbacks
            "unknown": "medium"
        }
        
        # Mapear tipo de documento
        if "type_classifier" in classifier_outputs:
            type_class = classifier_outputs["type_classifier"].get("predicted_class", "unknown")
            mapped_type = type_mapping.get(type_class, "announcement")
            mapped_outputs["document_type"] = {
                "predicted_class": mapped_type,
                "confidence": classifier_outputs["type_classifier"].get("confidence", 0.5),
                "probabilities": {}
            }
            
            # Mapear probabilidades
            if "probabilities" in classifier_outputs["type_classifier"]:
                for cls, prob in classifier_outputs["type_classifier"]["probabilities"].items():
                    mapped_cls = type_mapping.get(cls, cls)
                    mapped_outputs["document_type"]["probabilities"][mapped_cls] = prob
        
        # Mapear sentimento
        if "sentiment_analyzer" in classifier_outputs:
            sentiment_class = classifier_outputs["sentiment_analyzer"].get("predicted_class", "neutral")
            
            # Mapear "positivo"/"negativo" para "positive"/"negative" se necessário
            if sentiment_class == "positivo":
                sentiment_class = "positive"
            elif sentiment_class == "negativo":
                sentiment_class = "negative"
                
            # Ajustar para cinco níveis
            if sentiment_class == "positive" and classifier_outputs["sentiment_analyzer"].get("confidence", 0.5) > 0.8:
                mapped_sentiment = "very_positive"
            elif sentiment_class == "negative" and classifier_outputs["sentiment_analyzer"].get("confidence", 0.5) > 0.8:
                mapped_sentiment = "very_negative"
            else:
                mapped_sentiment = sentiment_mapping.get(sentiment_class, "neutral")
                
            mapped_outputs["sentiment"] = {
                "predicted_class": mapped_sentiment,
                "confidence": classifier_outputs["sentiment_analyzer"].get("confidence", 0.5),
                "probabilities": {}
            }
            
            # Mapear probabilidades
            if "probabilities" in classifier_outputs["sentiment_analyzer"]:
                for cls, prob in classifier_outputs["sentiment_analyzer"]["probabilities"].items():
                    mapped_cls = sentiment_mapping.get(cls, cls)
                    if cls == "positive" and prob > 0.8:
                        mapped_outputs["sentiment"]["probabilities"]["very_positive"] = prob * 0.7
                        mapped_outputs["sentiment"]["probabilities"]["positive"] = prob * 0.3
                    elif cls == "negative" and prob > 0.8:
                        mapped_outputs["sentiment"]["probabilities"]["very_negative"] = prob * 0.7
                        mapped_outputs["sentiment"]["probabilities"]["negative"] = prob * 0.3
                    else:
                        mapped_outputs["sentiment"]["probabilities"][mapped_cls] = prob
        
        # Mapear impacto
        if "impact_modeler" in classifier_outputs:
            impact_class = classifier_outputs["impact_modeler"].get("predicted_class", "medium")
            
            # Ajustar para cinco níveis
            if impact_class == "high" and classifier_outputs["impact_modeler"].get("confidence", 0.5) > 0.8:
                mapped_impact = "very_high"
            elif impact_class == "low" and classifier_outputs["impact_modeler"].get("confidence", 0.5) > 0.8:
                mapped_impact = "very_low"
            else:
                mapped_impact = impact_mapping.get(impact_class, "medium")
                
            mapped_outputs["impact"] = {
                "predicted_class": mapped_impact,
                "confidence": classifier_outputs["impact_modeler"].get("confidence", 0.5),
                "probabilities": {}
            }
            
            # Mapear probabilidades
            if "probabilities" in classifier_outputs["impact_modeler"]:
                for cls, prob in classifier_outputs["impact_modeler"]["probabilities"].items():
                    mapped_cls = impact_mapping.get(cls, cls)
                    if cls == "high" and prob > 0.8:
                        mapped_outputs["impact"]["probabilities"]["very_high"] = prob * 0.7
                        mapped_outputs["impact"]["probabilities"]["high"] = prob * 0.3
                    elif cls == "low" and prob > 0.8:
                        mapped_outputs["impact"]["probabilities"]["very_low"] = prob * 0.7
                        mapped_outputs["impact"]["probabilities"]["low"] = prob * 0.3
                    else:
                        mapped_outputs["impact"]["probabilities"][mapped_cls] = prob
                        
        return mapped_outputs
    
    
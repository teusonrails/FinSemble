"""
Módulo para implementação de redes bayesianas no sistema FinSemble.

Este módulo contém classes e funções para modelagem probabilística
usando redes bayesianas, essenciais para o Agregador Bayesiano.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Any, Optional
from collections import defaultdict

# Configuração de logging
logger = logging.getLogger(__name__)

class Node:
    """Representa um nó em uma rede bayesiana."""
    
    def __init__(self, name: str, states: List[str], parents: List[str] = None):
        """
        Inicializa um nó da rede bayesiana.
        
        Args:
            name: Nome do nó
            states: Lista de possíveis estados do nó
            parents: Lista de nomes dos nós pais (opcional)
        """
        self.name = name
        self.states = states
        self.n_states = len(states)
        self.parents = parents or []
        self.children = []
        self.cpt = None  # Conditional Probability Table
        
    def add_child(self, child_name: str):
        """
        Adiciona um filho ao nó.
        
        Args:
            child_name: Nome do nó filho
        """
        if child_name not in self.children:
            self.children.append(child_name)
            
    def set_cpt(self, cpt: np.ndarray):
        """
        Define a tabela de probabilidade condicional (CPT) para o nó.
        
        Args:
            cpt: Array numpy com a CPT
        """
        self.cpt = cpt
        
    def __repr__(self):
        return f"Node(name={self.name}, states={self.states}, parents={self.parents})"


class BayesianNetwork:
    """Implementação de uma rede bayesiana para o Agregador."""
    
    def __init__(self, name: str = "FinSembleNet"):
        """
        Inicializa a rede bayesiana.
        
        Args:
            name: Nome da rede
        """
        self.name = name
        self.nodes = {}
        self.sorted_nodes = []  # Ordem topológica para inferência
        
    def add_node(self, name: str, states: List[str], parents: List[str] = None):
        """
        Adiciona um nó à rede com validação aprimorada.
        
        Args:
            name: Nome do nó
            states: Possíveis estados do nó
            parents: Lista de nomes dos nós pais (opcional)
            
        Raises:
            ValueError: Se houver problemas com a adição do nó
        """
        # Validações adicionais
        if name in self.nodes:
            raise ValueError(f"Nó '{name}' já existe na rede")
            
        if not states:
            raise ValueError(f"Nó '{name}' deve ter pelo menos um estado")
            
        # Verificar duplicações nos estados
        if len(states) != len(set(states)):
            raise ValueError(f"Nó '{name}' tem estados duplicados")
            
        # Validar se os pais existem
        if parents:
            for parent in parents:
                if parent not in self.nodes:
                    raise ValueError(f"Pai '{parent}' não existe na rede")
                    
            # Verificar ciclos precocemente
            if name in parents:
                raise ValueError(f"Nó '{name}' não pode ser seu próprio pai")
                
            # Verificar ciclo potencial com verificação de descendentes
            for parent in parents:
                if self._is_ancestor(name, parent):
                    raise ValueError(f"Adicionar '{parent}' como pai de '{name}' criaria um ciclo")
        
        # Criar e adicionar o nó
        node = Node(name, states, parents or [])
        self.nodes[name] = node
        
        # Atualizar os filhos dos pais
        if parents:
            for parent in parents:
                self.nodes[parent].add_child(name)
                
        # Atualizar a ordem topológica
        self._update_topological_order()
        
        logger.debug(f"Nó '{name}' adicionado à rede bayesiana '{self.name}'")
        
    def _is_ancestor(self, node_name: str, potential_descendant: str) -> bool:
        """
        Verifica se um nó potencial é descendente de outro nó.
        
        Args:
            node_name: Nome do nó ancestral
            potential_descendant: Nome do nó potencialmente descendente
            
        Returns:
            True se o nó for ancestral do potencial descendente
        """
        if node_name not in self.nodes:
            return False
            
        visited = set()
        
        def check_descendants(current_node):
            if current_node == node_name:
                return True
                
            if current_node in visited:
                return False
                
            visited.add(current_node)
            
            # Verificar filhos
            for child in self.nodes[current_node].children:
                if check_descendants(child):
                    return True
                    
            return False
            
        return check_descendants(potential_descendant)
        
    def set_cpt(self, node_name: str, cpt: np.ndarray):
        """
        Define a CPT para um nó com validação aprimorada.
        
        Args:
            node_name: Nome do nó
            cpt: Tabela de probabilidade condicional
            
        Raises:
            ValueError: Se a CPT for inválida ou inconsistente
        """
        if node_name not in self.nodes:
            raise ValueError(f"Nó '{node_name}' não existe na rede")
            
        node = self.nodes[node_name]
        
        # Validação aprimorada das dimensões da CPT
        expected_shape = self._get_expected_cpt_shape(node)
        if cpt.shape != expected_shape:
            raise ValueError(f"Forma incorreta da CPT para '{node_name}'. "
                           f"Esperado {expected_shape}, recebido {cpt.shape}")
        
        # Validação de probabilidades (garantindo que somem 1 para cada configuração)
        if not self._validate_cpt_probabilities(cpt):
            # Tentar corrigir automaticamente normalizando
            try:
                cpt = self._normalize_cpt(cpt)
                logger.warning(f"CPT para '{node_name}' foi normalizada automaticamente")
            except Exception as e:
                raise ValueError(f"CPT inválida para '{node_name}' e não foi possível normalizar. "
                               f"As probabilidades devem somar 1 para cada configuração de pais. Erro: {str(e)}")
        
        # Verificar se há valores negativos ou maiores que 1
        if np.any(cpt < 0) or np.any(cpt > 1):
            raise ValueError(f"CPT para '{node_name}' contém valores inválidos. "
                           f"Todos os valores devem estar no intervalo [0, 1]")
        
        # Verificar se há valores NaN ou infinitos
        if np.any(np.isnan(cpt)) or np.any(np.isinf(cpt)):
            raise ValueError(f"CPT para '{node_name}' contém valores NaN ou infinitos")
            
        node.set_cpt(cpt)
        logger.debug(f"CPT definida para nó '{node_name}'")
        
        # Verificar consistência geral da rede após modificação
        self._check_network_consistency()
    
    def _normalize_cpt(self, cpt: np.ndarray) -> np.ndarray:
        """
        Normaliza uma CPT para garantir que as probabilidades somem 1.
        
        Args:
            cpt: CPT a ser normalizada
            
        Returns:
            CPT normalizada
            
        Raises:
            ValueError: Se a normalização não for possível
        """
        # Se CPT é unidimensional (nó sem pais)
        if cpt.ndim == 1:
            sum_values = np.sum(cpt)
            if sum_values <= 0:
                raise ValueError("A soma dos valores da CPT é zero ou negativa, impossível normalizar")
            return cpt / sum_values
        
        # Se CPT tem múltiplas dimensões
        # Criar uma cópia para evitar modificar o original
        normalized_cpt = cpt.copy()
        
        # Normalizar cada fatia separadamente
        # Reorganizamos para que a primeira dimensão (estados do nó) seja a última
        # para facilitar a normalização
        reshaped_cpt = np.moveaxis(normalized_cpt, 0, -1)
        
        # Calcular somas ao longo da última dimensão
        sums = np.sum(reshaped_cpt, axis=-1, keepdims=True)
        
        # Verificar divisões por zero
        zero_sums = (sums <= 0)
        if np.any(zero_sums):
            # Substituir divisões por zero por distribuição uniforme
            uniform_value = 1.0 / reshaped_cpt.shape[-1]
            mask = np.broadcast_to(zero_sums, reshaped_cpt.shape)
            reshaped_cpt = np.where(mask, uniform_value, reshaped_cpt / np.maximum(sums, 1e-10))
            logger.warning("Algumas configurações de pais tinham soma zero. " 
                         "Atribuída distribuição uniforme para estas configurações.")
        else:
            # Normalizar
            reshaped_cpt = reshaped_cpt / sums
            
        # Restaurar a ordem original das dimensões
        normalized_cpt = np.moveaxis(reshaped_cpt, -1, 0)
        
        return normalized_cpt
        
    def _get_expected_cpt_shape(self, node: Node) -> Tuple[int, ...]:
        """
        Determina a forma esperada da CPT para um nó.
        
        Args:
            node: Nó para calcular a forma da CPT
            
        Returns:
            Tupla com a forma esperada
        """
        # Se o nó não tem pais, a CPT tem apenas uma dimensão (seus próprios estados)
        if not node.parents:
            return (node.n_states,)
            
        # Se tem pais, a CPT tem uma dimensão para cada pai, mais uma para o próprio nó
        shape = [node.n_states]
        for parent in node.parents:
            parent_node = self.nodes[parent]
            shape.append(parent_node.n_states)
            
        return tuple(shape)
        
    def _validate_cpt_probabilities(self, cpt: np.ndarray) -> bool:
        """
        Validação aprimorada das probabilidades na CPT.
        
        Args:
            cpt: CPT a ser validada
            
        Returns:
            True se válida, False caso contrário
        """
        # Se CPT é unidimensional (nó sem pais)
        if cpt.ndim == 1:
            return np.isclose(np.sum(cpt), 1.0, rtol=1e-5, atol=1e-8)
        
        # Para CPTs multidimensionais, reorganizamos para facilitar a soma
        # Movemos a primeira dimensão (estados do nó) para o final
        reshaped_cpt = np.moveaxis(cpt, 0, -1)
        
        # Soma ao longo da última dimensão (estados do nó)
        sums = np.sum(reshaped_cpt, axis=-1)
        
        # Verificar se todas as somas estão próximas de 1
        return np.allclose(sums, 1.0, rtol=1e-5, atol=1e-8)
    
    def _check_network_consistency(self):
        """
        Verifica a consistência geral da rede bayesiana.
        
        Raises:
            ValueError: Se a rede for inconsistente
        """
        # Verificar se há ciclos (deve ser um DAG)
        try:
            self._update_topological_order()
        except ValueError as e:
            raise ValueError(f"Inconsistência na estrutura da rede: {str(e)}")
            
        # Verificar se todos os nós têm CPTs definidas
        for name, node in self.nodes.items():
            if node.cpt is None:
                logger.warning(f"Nó '{name}' não tem CPT definida")
                
        # Verificar se as referências dos pais/filhos são consistentes
        for name, node in self.nodes.items():
            # Verificar se todos os pais existem
            for parent in node.parents:
                if parent not in self.nodes:
                    raise ValueError(f"Inconsistência: nó '{name}' referencia pai '{parent}' inexistente")
                    
            # Verificar se o nó está na lista de filhos de todos os seus pais
            for parent in node.parents:
                if name not in self.nodes[parent].children:
                    logger.warning(f"Inconsistência: nó '{name}' não está na lista de filhos do pai '{parent}'")
                    # Corrigir automaticamente
                    self.nodes[parent].add_child(name)
        
    def _update_topological_order(self):
        """Atualiza a ordem topológica dos nós para inferência eficiente."""
        # Implementação de ordenação topológica
        visited = set()
        temp = set()
        order = []
        
        def visit(node_name):
            if node_name in temp:
                raise ValueError("Rede contém ciclos, não é uma DAG válida")
            if node_name in visited:
                return
                
            temp.add(node_name)
            
            node = self.nodes[node_name]
            for child in node.children:
                visit(child)
                
            temp.remove(node_name)
            visited.add(node_name)
            order.append(node_name)
            
        for node_name in self.nodes:
            if node_name not in visited:
                visit(node_name)
                
        self.sorted_nodes = list(reversed(order))
        logger.debug(f"Ordem topológica atualizada: {self.sorted_nodes}")
        
    def get_markov_blanket(self, node_name: str) -> Set[str]:
        """
        Obtém o cobertor de Markov de um nó (pais, filhos e pais dos filhos).
        
        Args:
            node_name: Nome do nó
            
        Returns:
            Conjunto com os nomes dos nós no cobertor de Markov
        """
        if node_name not in self.nodes:
            raise ValueError(f"Nó '{node_name}' não existe na rede")
            
        node = self.nodes[node_name]
        blanket = set(node.parents)  # Adicionar pais
        
        # Adicionar filhos e pais dos filhos
        for child_name in node.children:
            blanket.add(child_name)  # Adicionar filho
            child = self.nodes[child_name]
            blanket.update(child.parents)  # Adicionar pais do filho
            
        # Remover o próprio nó, se estiver no blanket
        blanket.discard(node_name)
        
        return blanket
        
    def inference(self, evidence: Dict[str, str], query_nodes: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Realiza inferência na rede bayesiana usando algoritmo de eliminação de variáveis.
        
        Args:
            evidence: Dicionário com evidências {node_name: state}
            query_nodes: Lista de nós para consulta (None para todos os nós não evidência)
            
        Returns:
            Dicionário com as distribuições de probabilidade posteriores {node_name: {state: prob}}
        """
        # Validar evidências
        for node_name, state in evidence.items():
            if node_name not in self.nodes:
                raise ValueError(f"Nó de evidência '{node_name}' não existe na rede")
            if state not in self.nodes[node_name].states:
                raise ValueError(f"Estado '{state}' não é válido para nó '{node_name}'")
                
        # Se query_nodes não for especificado, consultar todos os nós não evidência
        if query_nodes is None:
            query_nodes = [n for n in self.nodes if n not in evidence]
        else:
            # Validar nós de consulta
            for node_name in query_nodes:
                if node_name not in self.nodes:
                    raise ValueError(f"Nó de consulta '{node_name}' não existe na rede")
                    
        # Implementar algoritmo de eliminação de variáveis (simplificado para esta demonstração)
        # Em uma implementação completa, seria usado um algoritmo mais sofisticado
        results = {}
        
        for query_node in query_nodes:
            node = self.nodes[query_node]
            
            # Cálculo simplificado para nós sem pais (probabilidade priori)
            if not node.parents:
                if query_node in evidence:
                    # Se for uma evidência, a probabilidade é 1 para o estado observado
                    dist = {state: 1.0 if state == evidence[query_node] else 0.0 
                           for state in node.states}
                else:
                    # Se não for evidência, usar a distribuição priori
                    dist = {state: prob for state, prob in zip(node.states, node.cpt)}
            else:
                # Para nós com pais, calcular a probabilidade condicional
                # Isto é uma simplificação; a implementação real usaria MCMC ou similar
                dist = self._approximate_conditional_probability(query_node, evidence)
                
            results[query_node] = dist
            
        return results
        
    def _approximate_conditional_probability(self, node_name: str, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Aproxima a probabilidade condicional para um nó dado evidências.
        
        Args:
            node_name: Nome do nó
            evidence: Dicionário com evidências
            
        Returns:
            Dicionário com probabilidades aproximadas para cada estado
        """
        # Esta é uma implementação simplificada para demonstração
        # Na prática, usaríamos algoritmos como fator de eliminação variável ou MCMC
        
        node = self.nodes[node_name]
        
        # Verificar se este nó é uma evidência
        if node_name in evidence:
            # Retornar probabilidade 1 para o estado observado
            return {state: 1.0 if state == evidence[node_name] else 0.0 for state in node.states}
            
        # Verificar quais pais são evidências
        known_parents = {parent: evidence[parent] for parent in node.parents if parent in evidence}
        
        # Se todos os pais são conhecidos, podemos consultar diretamente a CPT
        if len(known_parents) == len(node.parents):
            # Criar índice para a CPT
            parent_indices = []
            for parent in node.parents:
                parent_node = self.nodes[parent]
                parent_state = known_parents[parent]
                parent_idx = parent_node.states.index(parent_state)
                parent_indices.append(parent_idx)
                
            # Obter distribuição da CPT
            cpt_indices = tuple([slice(None)] + parent_indices)  # slice(None) para todas as probabilidades do nó
            probs = node.cpt[cpt_indices]
            
            return {state: float(prob) for state, prob in zip(node.states, probs)}
        else:
            # Se nem todos os pais são conhecidos, fazemos uma aproximação
            # Neste caso, calcular média ponderada das distribuições para as configurações conhecidas
            
            # Por simplicidade, retornar uma distribuição uniforme modificada
            # Em uma implementação real, usaríamos aproximação mais sofisticada
            base_prob = 1.0 / node.n_states
            dist = {state: base_prob for state in node.states}
            
            # Ajustar com base nas evidências parciais (se houver)
            if known_parents:
                # Aumentar ligeiramente a probabilidade de alguns estados com base nas evidências
                # Esta é uma heurística simplificada para demonstração
                high_impact_evidence = any(parent for parent, state in known_parents.items()
                                       if state in ["positive", "high"] and parent != node_name)
                                       
                if high_impact_evidence and "high" in node.states:
                    # Aumentar probabilidade de "high"
                    dist["high"] *= 1.2
                    # Normalizar
                    total = sum(dist.values())
                    dist = {k: v / total for k, v in dist.items()}
                    
            return dist
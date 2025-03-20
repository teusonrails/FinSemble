"""
Módulo para algoritmos de propagação de crenças em redes bayesianas.

Este módulo implementa algoritmos para inferência probabilística em
redes bayesianas, incluindo propagação de crenças e junção de árvores.
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict

from utils.bayesian_network import BayesianNetwork

# Configuração de logging
logger = logging.getLogger(__name__)

class Factor:
    """Representa um fator em uma rede bayesiana."""
    
    def __init__(self, variables: List[str], values: np.ndarray, 
                variable_domains: Dict[str, List[str]]):
        """
        Inicializa um fator.
        
        Args:
            variables: Lista de variáveis no fator
            values: Array numpy com valores do fator
            variable_domains: Domínios (possíveis valores) das variáveis
        """
        self.variables = variables
        self.values = values
        self.variable_domains = variable_domains
        
    def marginalize(self, variable: str) -> 'Factor':
        """
        Marginaliza uma variável do fator.
        
        Args:
            variable: Variável a ser marginalizada
            
        Returns:
            Novo fator marginalizado
        """
        if variable not in self.variables:
            return self
            
        # Encontrar índice da variável
        var_idx = self.variables.index(variable)
        
        # Criar nova lista de variáveis excluindo a variável marginalizada
        new_variables = [v for v in self.variables if v != variable]
        
        if not new_variables:
            # Se não sobram variáveis, retornar escalar
            return Factor([], np.sum(self.values), self.variable_domains)
            
        # Marginalizar somando ao longo da dimensão da variável
        new_values = np.sum(self.values, axis=var_idx)
        
        return Factor(new_variables, new_values, self.variable_domains)
        
    def product(self, other: 'Factor') -> 'Factor':
        """
        Calcula o produto de dois fatores com otimizações.
        
        Args:
            other: Outro fator
            
        Returns:
            Produto dos fatores
        """
        # Otimização: se um dos fatores tem zero variáveis, é um escalar
        if not self.variables:
            return Factor(other.variables, self.values * other.values, self.variable_domains)
            
        if not other.variables:
            return Factor(self.variables, self.values * other.values, self.variable_domains)
        
        # Otimização: verificar quais variáveis são compartilhadas
        shared_vars = set(self.variables).intersection(other.variables)
        only_in_self = [v for v in self.variables if v not in shared_vars]
        only_in_other = [v for v in other.variables if v not in shared_vars]
        
        # Determinar conjunto de variáveis para o produto final
        new_variables = list(shared_vars) + only_in_self + only_in_other
        
        # Preparar índices para broadcast
        self_indices = [new_variables.index(var) for var in self.variables]
        other_indices = [new_variables.index(var) for var in other.variables]
        
        # Calcular as dimensões para cada variável
        var_to_dim = {var: len(self.variable_domains[var]) for var in new_variables}
        
        # Criar formatos para broadcast
        self_shape = [1] * len(new_variables)
        other_shape = [1] * len(new_variables)
        
        for i, idx in enumerate(self_indices):
            dim = var_to_dim[self.variables[i]]
            self_shape[idx] = dim
            
        for i, idx in enumerate(other_indices):
            dim = var_to_dim[other.variables[i]]
            other_shape[idx] = dim
            
        # Otimização: redimensionar valores para broadcast
        try:
            self_reshaped = np.reshape(self.values, self_shape)
            other_reshaped = np.reshape(other.values, other_shape)
        except ValueError:
            # Fallback para método anterior se o reshape falhar
            self_reshaped = self.values.reshape(self_shape)
            other_reshaped = other.values.reshape(other_shape)
        
        # Multiplicar e criar novo fator
        new_values = self_reshaped * other_reshaped
        
        return Factor(new_variables, new_values, self.variable_domains)
        
    def normalize(self) -> 'Factor':
        """
        Normaliza o fator para que seus valores somem 1.
        
        Returns:
            Novo fator normalizado
        """
        total = np.sum(self.values)
        if total == 0:
            # Evitar divisão por zero
            new_values = np.ones_like(self.values) / np.size(self.values)
        else:
            new_values = self.values / total
            
        return Factor(self.variables.copy(), new_values, self.variable_domains)

    @staticmethod
    def from_cpt(node_name: str, states: List[str], parents: List[str], 
                cpt: np.ndarray, variable_domains: Dict[str, List[str]]) -> 'Factor':
        """
        Cria um fator a partir de uma tabela de probabilidade condicional (CPT).
        
        Args:
            node_name: Nome do nó
            states: Estados do nó
            parents: Pais do nó
            cpt: CPT do nó
            variable_domains: Domínios das variáveis
            
        Returns:
            Fator correspondente à CPT
        """
        variables = [node_name] + parents
        values = cpt
        
        # Se não houver pais, a CPT é unidimensional
        if not parents:
            return Factor([node_name], values, variable_domains)
        
        # Verificar se a CPT precisa ser reorganizada
        # A CPT tem dimensões [estados_nó, estados_pai1, estados_pai2, ...]
        # Se necessário, reorganizar para ajustar à ordem das variáveis
        if variables[0] != node_name:
            # Reorganizar dimensões para corresponder à ordem das variáveis
            desired_order = [variables.index(node_name)] + list(range(1, len(variables)))
            if desired_order != list(range(len(variables))):
                values = np.transpose(values, desired_order)
        
        return Factor(variables, values, variable_domains)

class BeliefPropagator:
    """Implementa algoritmos de propagação de crenças em redes bayesianas."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o propagador de crenças.
        
        Args:
            config: Configurações para o propagador
        """
        self.config = config or {}
        self.max_iterations = config.get("max_iterations", 100)
        self.convergence_threshold = config.get("convergence_threshold", 1e-6)
        self.damping = config.get("damping", 0.5)
        
    def variable_elimination(self, factors: List[Factor], 
                       query_variables: List[str], 
                       elimination_order: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """
        Implementa o algoritmo de eliminação de variáveis otimizado.
        
        Args:
            factors: Lista de fatores
            query_variables: Variáveis de consulta
            elimination_order: Ordem de eliminação (opcional)
            
        Returns:
            Dicionário com distribuições de probabilidade para variáveis de consulta
        """
        if not factors:
            logger.warning("Lista de fatores vazia")
            return {var: {} for var in query_variables}
            
        # Evitar modificar a lista original
        current_factors = list(factors)
        
        # Obter todas as variáveis
        all_variables = set()
        for factor in factors:
            all_variables.update(factor.variables)
            
        # Determinar variáveis a eliminar (excluindo variáveis de consulta)
        eliminate_variables = list(all_variables - set(query_variables))
        
        # Determinar a ordem de eliminação
        if elimination_order is None:
            # Heurística min-fill: escolhe a variável que adiciona menos arestas ao grafo moral
            elimination_order = self._min_fill_heuristic(factors, eliminate_variables, query_variables)
        else:
            # Verificar se a ordem inclui todas as variáveis a eliminar
            missing_vars = set(eliminate_variables) - set(elimination_order)
            if missing_vars:
                logger.warning(f"Variáveis ausentes na ordem de eliminação: {missing_vars}")
                # Adicionar variáveis faltantes ao final da ordem
                elimination_order = elimination_order + list(missing_vars)
        
        # Eliminar variáveis uma a uma
        for var in elimination_order:
            # Pular variáveis de consulta
            if var in query_variables:
                continue
                
            # Encontrar fatores relevantes de forma eficiente
            relevant_factors = []
            irrelevant_factors = []
            
            for factor in current_factors:
                if var in factor.variables:
                    relevant_factors.append(factor)
                else:
                    irrelevant_factors.append(factor)
                    
            if not relevant_factors:
                continue
                
            # Multiplicar fatores relevantes de forma otimizada
            # Começar com o menor fator e multiplicar progressivamente
            relevant_factors.sort(key=lambda f: f.values.size)
            product_factor = relevant_factors[0]
            
            for i in range(1, len(relevant_factors)):
                product_factor = product_factor.product(relevant_factors[i])
                
            # Marginalizar a variável
            marginal_factor = product_factor.marginalize(var)
            
            # Atualizar lista de fatores
            current_factors = irrelevant_factors + [marginal_factor]
        
        # Juntar todos os fatores finais
        if len(current_factors) > 1:
            # Ordenar por tamanho para eficiência
            current_factors.sort(key=lambda f: f.values.size)
            final_factor = current_factors[0]
            
            for i in range(1, len(current_factors)):
                final_factor = final_factor.product(current_factors[i])
        elif len(current_factors) == 1:
            final_factor = current_factors[0]
        else:
            # Caso sem fatores (não deve ocorrer)
            return {var: {} for var in query_variables}
        
        # Obter distribuições marginais para variáveis de consulta
        results = {}
        
        for var in query_variables:
            # Se a variável não estiver no fator final
            if var not in final_factor.variables:
                # Distribuição uniforme
                states = factors[0].variable_domains[var]
                results[var] = {state: 1.0/len(states) for state in states}
                continue
                
            # Se houver outras variáveis, marginalizar todas exceto a atual
            temp_factor = final_factor
            for other_var in final_factor.variables:
                if other_var != var:
                    temp_factor = temp_factor.marginalize(other_var)
                    
            # Normalizar
            temp_factor = temp_factor.normalize()
            
            # Converter para dicionário
            states = factors[0].variable_domains[var]
            results[var] = {state: float(prob) for state, prob in zip(states, temp_factor.values)}
        
        return results
    
    def _min_fill_heuristic(self, factors: List[Factor], 
                       eliminate_variables: List[str],
                       query_variables: List[str]) -> List[str]:
        """
        Implementa a heurística min-fill para escolher a ordem de eliminação de variáveis.
        
        Esta heurística escolhe a variável que adiciona o menor número de arestas ao grafo.
        
        Args:
            factors: Lista de fatores
            eliminate_variables: Variáveis a eliminar
            query_variables: Variáveis de consulta
            
        Returns:
            Ordem otimizada de eliminação
        """
        # Construir grafo não direcionado representando dependências entre variáveis
        graph = {}
        
        # Adicionar nós
        all_vars = set(eliminate_variables) | set(query_variables)
        for var in all_vars:
            graph[var] = set()
        
        # Adicionar arestas
        for factor in factors:
            vars_in_factor = factor.variables
            for i in range(len(vars_in_factor)):
                for j in range(i+1, len(vars_in_factor)):
                    graph[vars_in_factor[i]].add(vars_in_factor[j])
                    graph[vars_in_factor[j]].add(vars_in_factor[i])
        
        # Calcular ordem de eliminação usando min-fill
        ordering = []
        remaining_vars = set(eliminate_variables)
        
        while remaining_vars:
            # Encontrar variável que adiciona menos arestas quando eliminada
            min_fill_var = None
            min_fill_count = float('inf')
            
            for var in remaining_vars:
                # Vizinhos não conectados
                neighbors = graph[var]
                fill_edges = 0
                
                # Contar quantas arestas seriam adicionadas
                for i, neigh1 in enumerate(neighbors):
                    for neigh2 in list(neighbors)[i+1:]:
                        if neigh2 not in graph[neigh1]:
                            fill_edges += 1
                
                if fill_edges < min_fill_count:
                    min_fill_count = fill_edges
                    min_fill_var = var
            
            # Adicionar à ordenação
            ordering.append(min_fill_var)
            remaining_vars.remove(min_fill_var)
            
            # Atualizar o grafo (adicionar arestas entre vizinhos)
            neighbors = list(graph[min_fill_var])
            for i in range(len(neighbors)):
                for j in range(i+1, len(neighbors)):
                    graph[neighbors[i]].add(neighbors[j])
                    graph[neighbors[j]].add(neighbors[i])
                    
            # Remover o nó do grafo
            for neigh in neighbors:
                if min_fill_var in graph[neigh]:
                    graph[neigh].remove(min_fill_var)
            del graph[min_fill_var]
        
        return ordering        
    
    def loopy_belief_propagation(self, bayesian_network: 'BayesianNetwork',
                           evidence: Dict[str, str],
                           query_nodes: List[str] = None,
                           max_iterations: int = None) -> Dict[str, Dict[str, float]]:
        """
        Implementa o algoritmo de propagação de crenças com loops (Loopy Belief Propagation).
        
        Args:
            bayesian_network: Rede bayesiana
            evidence: Evidências observadas
            query_nodes: Nós para consulta (None para todos os nós não evidência)
            max_iterations: Máximo de iterações (None para usar configuração global)
            
        Returns:
            Distribuições marginais para os nós de consulta
        """
        if max_iterations is None:
            max_iterations = self.max_iterations
            
        # Determinar nós de consulta
        if query_nodes is None:
            query_nodes = [n for n in bayesian_network.nodes if n not in evidence]
            
        # Criar fatores a partir da rede bayesiana
        factors = []
        variable_domains = {}
        
        # Mapear variáveis para fatores e vice-versa
        var_to_factors = defaultdict(list)
        factor_to_vars = {}
        
        # Converter a rede para fatores
        for node_name, node in bayesian_network.nodes.items():
            # Registrar domínios das variáveis
            variable_domains[node_name] = node.states
            
            # Criar fator a partir da CPT
            factor_vars = [node_name] + node.parents
            factor_name = f"f_{node_name}"
            
            # Registrar mapeamentos
            factor_to_vars[factor_name] = factor_vars
            for var in factor_vars:
                var_to_factors[var].append(factor_name)
                
            # Criar e adicionar o fator
            factor = Factor.from_cpt(
                node_name, 
                node.states, 
                node.parents, 
                node.cpt, 
                variable_domains
            )
            factors.append((factor_name, factor))
        
        # Incorporar evidências como fatores adicionais
        for var, val in evidence.items():
            if var not in variable_domains:
                continue
                
            # Criar um fator unitário para a evidência
            var_idx = variable_domains[var].index(val)
            evidence_values = np.zeros(len(variable_domains[var]))
            evidence_values[var_idx] = 1.0
            
            evidence_factor = Factor([var], evidence_values, variable_domains)
            
            # Registrar fator de evidência
            evidence_factor_name = f"f_ev_{var}"
            factors.append((evidence_factor_name, evidence_factor))
            factor_to_vars[evidence_factor_name] = [var]
            var_to_factors[var].append(evidence_factor_name)
        
        # Inicializar mensagens (var->factor e factor->var)
        messages = {}
        factor_dict = dict(factors)
        
        # Para cada variável e cada fator conectado
        for var in var_to_factors:
            for factor_name in var_to_factors[var]:
                # Mensagem variável -> fator (inicialmente uniforme)
                domain_size = len(variable_domains[var])
                messages[(var, factor_name)] = np.ones(domain_size) / domain_size
                
                # Mensagem fator -> variável (inicialmente uniforme)
                messages[(factor_name, var)] = np.ones(domain_size) / domain_size
        
        # Algoritmo de passagem de mensagens
        converged = False
        
        for iteration in range(max_iterations):
            old_messages = {k: v.copy() for k, v in messages.items()}
            max_diff = 0.0
            
            # 1. Atualizar mensagens: variável -> fator
            for var in var_to_factors:
                domain_size = len(variable_domains[var])
                
                for factor_name in var_to_factors[var]:
                    # Iniciar com mensagem uniforme
                    msg = np.ones(domain_size)
                    
                    # Multiplicar por todas as mensagens de outros fatores para esta variável
                    for other_factor in var_to_factors[var]:
                        if other_factor != factor_name:
                            msg *= messages[(other_factor, var)]
                    
                    # Normalizar a mensagem
                    msg_sum = np.sum(msg)
                    if msg_sum > 0:
                        msg = msg / msg_sum
                    else:
                        msg = np.ones(domain_size) / domain_size
                    
                    # Atualizar a mensagem no dicionário
                    messages[(var, factor_name)] = msg
            
            # 2. Atualizar mensagens: fator -> variável
            for factor_name, factor in factors:
                factor_vars = factor_to_vars[factor_name]
                
                for var_idx, var in enumerate(factor_vars):
                    # Domínio da variável
                    domain_size = len(variable_domains[var])
                    
                    # Inicializar mensagem
                    msg = np.zeros(domain_size)
                    
                    # Para cada possível valor da variável atual
                    for val_idx in range(domain_size):
                        # Criar um dicionário para armazenar a atribuição atual
                        var_assignment = {var: val_idx}
                        
                        # Calcular a soma sobre todas as outras variáveis
                        self._sum_product_message(
                            factor_name, factor, factor_vars, var_idx, 
                            var_assignment, messages, variable_domains, 0, 1.0, msg
                        )
                    
                    # Normalizar a mensagem
                    msg_sum = np.sum(msg)
                    if msg_sum > 0:
                        msg = msg / msg_sum
                    else:
                        msg = np.ones(domain_size) / domain_size
                    
                    # Atualizar a mensagem e calcular a diferença
                    old_msg = old_messages.get((factor_name, var), np.zeros(domain_size))
                    diff = np.max(np.abs(msg - old_msg))
                    max_diff = max(max_diff, diff)
                    
                    # Aplicar damping para ajudar na convergência
                    if self.damping > 0:
                        msg = self.damping * old_msg + (1 - self.damping) * msg
                    
                    messages[(factor_name, var)] = msg
            
            # Verificar convergência
            if max_diff < self.convergence_threshold:
                converged = True
                logger.info(f"LBP convergiu após {iteration+1} iterações")
                break
        
        if not converged:
            logger.warning(f"LBP não convergiu após {max_iterations} iterações")
        
        # Calcular crenças marginais para os nós de consulta
        beliefs = {}
        
        for var in query_nodes:
            if var not in var_to_factors:
                continue
                
            # Inicializar com vetor uniforme
            domain_size = len(variable_domains[var])
            belief = np.ones(domain_size)
            
            # Multiplicar por todas as mensagens para esta variável
            for factor_name in var_to_factors[var]:
                belief *= messages[(factor_name, var)]
            
            # Normalizar
            belief_sum = np.sum(belief)
            if belief_sum > 0:
                belief = belief / belief_sum
            else:
                belief = np.ones(domain_size) / domain_size
            
            # Converter para dicionário
            beliefs[var] = {
                state: float(prob) for state, prob in zip(variable_domains[var], belief)
            }
        
        return beliefs

    def _sum_product_message(self, factor_name, factor, factor_vars, target_var_idx, 
                        var_assignment, messages, variable_domains, 
                        current_var_idx, factor_value, result_msg):
        """
        Função auxiliar recursiva para cálculo de mensagens no algoritmo sum-product.
        
        Calcula a mensagem de um fator para uma variável percorrendo todas as 
        possíveis atribuições de valores para as outras variáveis.
        """
        if current_var_idx == len(factor_vars):
            # Extrair o índice da variável alvo na atribuição atual
            target_var = factor_vars[target_var_idx]
            target_val_idx = var_assignment[target_var]
            
            # Multiplicar pelo valor do fator para esta atribuição e acumular no resultado
            result_msg[target_val_idx] += factor_value
            return
        
        # Pular a variável alvo na recursão
        if current_var_idx == target_var_idx:
            self._sum_product_message(
                factor_name, factor, factor_vars, target_var_idx,
                var_assignment, messages, variable_domains,
                current_var_idx + 1, factor_value, result_msg
            )
            return
        
        # Para cada valor possível da variável atual
        var = factor_vars[current_var_idx]
        domain_size = len(variable_domains[var])
        
        for val_idx in range(domain_size):
            # Atualizar a atribuição para esta variável
            var_assignment[var] = val_idx
            
            # Obter a mensagem da variável para este fator
            msg_value = messages.get((var, factor_name), np.ones(domain_size))[val_idx]
            
            # Continuar a recursão
            self._sum_product_message(
                factor_name, factor, factor_vars, target_var_idx,
                var_assignment, messages, variable_domains,
                current_var_idx + 1, factor_value * msg_value, result_msg
            )
    
    def junction_tree_inference(self, bayesian_network: 'BayesianNetwork', 
                             evidence: Dict[str, str],
                             query_nodes: List[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Implementa o algoritmo de árvore de junção para inferência.
        Este é mais eficiente que eliminação de variáveis para múltiplas consultas.
        
        Args:
            bayesian_network: Rede bayesiana
            evidence: Evidências observadas
            query_nodes: Nós para consulta (None para todos os nós não evidência)
            
        Returns:
            Distribuições marginais para os nós de consulta
        """
        # Implementação simplificada do algoritmo de árvore de junção
        # Em uma implementação real, seria mais completo
        
        # Convertemos a rede bayesiana para fatores
        factors = []
        variable_domains = {}
        
        for node_name, node in bayesian_network.nodes.items():
            # Registrar domínios das variáveis
            variable_domains[node_name] = node.states
            
            # Criar fator a partir da CPT
            factor = Factor.from_cpt(
                node_name, 
                node.states, 
                node.parents, 
                node.cpt, 
                variable_domains
            )
            
            # Incorporar evidências, se houver
            if node_name in evidence:
                # Criar fator de evidência
                evidence_idx = node.states.index(evidence[node_name])
                evidence_values = np.zeros(len(node.states))
                evidence_values[evidence_idx] = 1.0
                evidence_factor = Factor([node_name], evidence_values, variable_domains)
                
                # Multiplicar pelo fator original
                factor = factor.product(evidence_factor)
                
            factors.append(factor)
        
        # Determinar nós de consulta
        if query_nodes is None:
            query_nodes = [n for n in bayesian_network.nodes if n not in evidence]
        
        # Para simplicidade, usamos eliminação de variáveis
        # Em uma implementação completa, construiríamos a árvore de junção
        return self.variable_elimination(factors, query_nodes)
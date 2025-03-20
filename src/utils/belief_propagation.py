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
        new_variables = self.variables.copy()
        new_variables.remove(variable)
        
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
        Marginaliza uma variável do fator com verificações aprimoradas.
        
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
            
        # Otimização: usar einsum para marginalização eficiente
        if var_idx == 0:
            # Caso especial para eficiência
            new_values = np.sum(self.values, axis=0)
        elif var_idx == len(self.variables) - 1:
            # Outro caso especial para eficiência
            new_values = np.sum(self.values, axis=-1)
        else:
            # Caso geral
            # Construir string para einsum
            input_idx = list(range(len(self.variables)))
            output_idx = [i for i in input_idx if i != var_idx]
            
            # Usar einsum para marginalização eficiente
            new_values = np.einsum(self.values, input_idx, output_idx)
            
        return Factor(new_variables, new_values, self.variable_domains)

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
        current_factors = factors.copy()
        
        # Obter todas as variáveis
        all_variables = set()
        for factor in factors:
            all_variables.update(factor.variables)
            
        # Determinar variáveis a eliminar
        eliminate_variables = list(all_variables - set(query_variables))
        
        # Determinar a ordem de eliminação
        if elimination_order is None:
            # Heurística: eliminar variáveis que aparecem em menos fatores primeiro
            var_count = {var: sum(1 for f in current_factors if var in f.variables) 
                        for var in eliminate_variables}
            elimination_order = sorted(eliminate_variables, key=lambda v: var_count[v])
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
                
            # Otimização: encontrar fatores relevantes de forma eficiente
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
            product_factor = relevant_factors[0]
            for i in range(1, len(relevant_factors)):
                product_factor = product_factor.product(relevant_factors[i])
                
            # Marginalizar a variável
            marginal_factor = product_factor.marginalize(var)
            
            # Atualizar lista de fatores
            current_factors = irrelevant_factors + [marginal_factor]
            
        # Preparar resultados para cada variável de consulta
        results = {}
        variable_factors = {var: [] for var in query_variables}
        
        # Agrupar fatores por variável de consulta
        for factor in current_factors:
            for var in factor.variables:
                if var in query_variables:
                    variable_factors[var].append(factor)
                    
        # Calcular distribuição para cada variável de consulta
        for var in query_variables:
            if not variable_factors[var]:
                # Se não há fatores para esta variável, usar distribuição uniforme
                states = factors[0].variable_domains[var]
                results[var] = {state: 1.0/len(states) for state in states}
                continue
                
            # Multiplicar todos os fatores relevantes
            result_factor = variable_factors[var][0]
            for i in range(1, len(variable_factors[var])):
                result_factor = result_factor.product(variable_factors[var][i])
                
            # Marginalizar todas as variáveis exceto a de consulta
            for other_var in result_factor.variables:
                if other_var != var:
                    result_factor = result_factor.marginalize(other_var)
                    
            # Normalizar
            total = np.sum(result_factor.values)
            if total > 0:
                normalized_values = result_factor.values / total
            else:
                # Distribuição uniforme se soma for zero
                normalized_values = np.ones_like(result_factor.values) / len(result_factor.values)
                
            # Converter para dicionário
            states = factors[0].variable_domains[var]
            results[var] = {state: float(prob) for state, prob in zip(states, normalized_values)}
            
        return results
            
    def loopy_belief_propagation(self, bayesian_network: 'BayesianNetwork',
                               evidence: Dict[str, str],
                               query_nodes: List[str] = None,
                               max_iterations: int = None) -> Dict[str, Dict[str, float]]:
        """
        Implementa o algoritmo de propagação de crenças com loops melhorado.
        
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
            
        # Construir grafo de fatores
        factors = []
        variable_domains = {}
        factor_to_vars = {}
        var_to_factors = defaultdict(list)
        
        # Converter a rede bayesiana para fatores
        for i, (node_name, node) in enumerate(bayesian_network.nodes.items()):
            # Registrar domínios
            variable_domains[node_name] = node.states
            
            # Criar fator a partir da CPT
            factor = Factor.from_cpt(
                node_name, 
                node.states, 
                node.parents, 
                node.cpt, 
                variable_domains
            )
            
            # Registrar mapeamentos de fator para variáveis e vice-versa
            factor_name = f"f{i}"
            factor_to_vars[factor_name] = factor.variables
            
            for var in factor.variables:
                var_to_factors[var].append(factor_name)
                
            factors.append(factor)
            
        # Incorporar evidências
        for var, val in evidence.items():
            # Criar fator de evidência
            var_idx = bayesian_network.nodes[var].states.index(val)
            evidence_values = np.zeros(len(bayesian_network.nodes[var].states))
            evidence_values[var_idx] = 1.0
            
            evidence_factor = Factor([var], evidence_values, variable_domains)
            factors.append(evidence_factor)
            
            # Atualizar mapeamentos
            factor_name = f"f{len(factors)-1}"
            factor_to_vars[factor_name] = [var]
            var_to_factors[var].append(factor_name)
        
        # Inicializar mensagens
        messages = {}
        for var in var_to_factors:
            for factor_name in var_to_factors[var]:
                # Mensagem variável -> fator
                messages[(var, factor_name)] = np.ones(len(variable_domains[var]))
                
                # Mensagem fator -> variável
                messages[(factor_name, var)] = np.ones(len(variable_domains[var]))
                
        # Algoritmo de passagem de mensagens (Sum-Product)
        for iteration in range(max_iterations):
            old_messages = messages.copy()
            
            # Atualizar mensagens variável -> fator
            for var in var_to_factors:
                for factor_name in var_to_factors[var]:
                    # Produto das mensagens de outros fatores para esta variável
                    msg = np.ones(len(variable_domains[var]))
                    for other_factor in var_to_factors[var]:
                        if other_factor != factor_name:
                            msg *= messages[(other_factor, var)]
                            
                    messages[(var, factor_name)] = msg
                    
            # Atualizar mensagens fator -> variável
            for factor_idx, factor in enumerate(factors):
                factor_name = f"f{factor_idx}"
                vars_in_factor = factor_to_vars[factor_name]
                
                for var in vars_in_factor:
                    # Marginalização do produto do fator com mensagens de outras variáveis
                    # Cálculo completo seria mais complexo, mas simplificamos para ilustração
                    
                    # Na prática, calcularíamos:
                    # 1. Produto do fator com mensagens de outras variáveis
                    # 2. Marginalização para obter mensagem para a variável atual
                    
                    # Implementação simplificada
                    msg = np.ones(len(variable_domains[var]))
                    var_idx = factor.variables.index(var) if var in factor.variables else -1
                    
                    if var_idx >= 0 and var_idx < factor.values.ndim:
                        # Marginalizar ao longo do eixo da variável
                        msg = np.sum(factor.values, axis=var_idx)
                        
                    messages[(factor_name, var)] = msg
                    
            # Verificar convergência
            max_diff = 0.0
            for key in messages:
                diff = np.max(np.abs(messages[key] - old_messages.get(key, np.zeros_like(messages[key]))))
                max_diff = max(max_diff, diff)
                
            if max_diff < self.convergence_threshold:
                logger.info(f"Convergência alcançada na iteração {iteration+1}")
                break
                
        # Calcular crenças marginais
        beliefs = {}
        for var in query_nodes:
            if var in var_to_factors:
                # Produto de todas as mensagens para esta variável
                belief = np.ones(len(variable_domains[var]))
                for factor_name in var_to_factors[var]:
                    belief *= messages[(factor_name, var)]
                    
                # Normalizar
                belief = belief / np.sum(belief)
                
                # Converter para dicionário
                beliefs[var] = {state: float(prob) 
                              for state, prob in zip(variable_domains[var], belief)}
                              
        return beliefs
    
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
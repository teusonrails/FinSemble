"""
Módulo de utilitários de performance para o sistema FinSemble.

Este módulo fornece ferramentas e técnicas para otimizar o desempenho
de diferentes componentes do sistema, especialmente para processamento
de grandes volumes de dados.
"""

import time
import logging
import functools
import numpy as np
from typing import Dict, List, Any, Callable, TypeVar, cast

# Configuração de logging
logger = logging.getLogger(__name__)

# Definição de tipos para cache
T = TypeVar('T')
CacheFunction = Callable[..., T]

def timed(func: Callable) -> Callable:
    """
    Decorador para medir o tempo de execução de uma função.
    
    Args:
        func: Função a ser monitorada
        
    Returns:
        Função decorada com medição de tempo
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Função {func.__name__} executada em {elapsed_time:.4f} segundos")
        return result
    return wrapper

def cached_feature_extraction(maxsize: int = 128) -> Callable[[CacheFunction[T]], CacheFunction[T]]:
    """
    Decorador para cache de resultados de extração de características.
    
    Args:
        maxsize: Tamanho máximo do cache
        
    Returns:
        Decorador configurado
    """
    def decorator(func: CacheFunction[T]) -> CacheFunction[T]:
        cache = functools.lru_cache(maxsize=maxsize)(func)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extrair o texto do primeiro argumento (normalmente self) ou de kwargs
            text = kwargs.get('text', None)
            if text is None and len(args) > 1:
                text = args[1]
                
            # Logging para depuração de cache
            if hasattr(cache, 'cache_info'):
                info = cache.cache_info()
                hit_ratio = info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0
                logger.debug(f"Cache para {func.__name__}: {hit_ratio:.2%} de acertos ({info.hits} acertos, {info.misses} falhas)")
                
            return cache(*args, **kwargs)
            
        # Adicionar método para limpar cache
        wrapper.clear_cache = cache.cache_clear  # type: ignore
        return cast(CacheFunction[T], wrapper)
        
    return decorator

def batch_process(items: List[Any], 
                 process_func: Callable[[Any], Any], 
                 batch_size: int = 100, 
                 parallel: bool = False,
                 n_jobs: int = -1) -> List[Any]:
    """
    Processa uma lista de itens em lotes, opcionalmente em paralelo.
    
    Args:
        items: Lista de itens para processar
        process_func: Função para processar cada item
        batch_size: Tamanho do lote
        parallel: Se deve processar em paralelo
        n_jobs: Número de jobs para processamento paralelo (-1 para todos os CPUs)
        
    Returns:
        Lista de resultados processados
    """
    if not items:
        return []
        
    results = []
    n_batches = (len(items) + batch_size - 1) // batch_size
    
    # Processar em paralelo se solicitado
    if parallel:
        try:
            from joblib import Parallel, delayed
            n_jobs = n_jobs if n_jobs > 0 else None  # None significa todos os CPUs no joblib
            
            # Dividir em lotes para processamento paralelo
            batches = [items[i * batch_size:(i + 1) * batch_size] for i in range(n_batches)]
            
            # Função para processar um lote
            def process_batch(batch):
                return [process_func(item) for item in batch]
                
            # Processar lotes em paralelo
            batch_results = Parallel(n_jobs=n_jobs)(delayed(process_batch)(batch) for batch in batches)
            
            # Combinar resultados
            for batch in batch_results:
                results.extend(batch)
                
        except ImportError:
            logger.warning("Joblib não disponível. Processando sequencialmente.")
            # Fallback para processamento sequencial
            for item in items:
                results.append(process_func(item))
    else:
        # Processamento sequencial em lotes
        for i in range(n_batches):
            batch = items[i * batch_size:(i + 1) * batch_size]
            for item in batch:
                results.append(process_func(item))
                
    return results

def optimize_array_operations(func: Callable) -> Callable:
    """
    Decorador para otimizar operações em arrays.
    
    Args:
        func: Função a ser otimizada
        
    Returns:
        Função decorada com otimizações
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Configuração do numpy para otimização
        old_settings = np.seterr(all='ignore')  # Evitar avisos durante operações
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restaurar configurações
            np.seterr(**old_settings)
            
    return wrapper
"""
Módulo para processamento paralelo de textos no FinSemble.

Este módulo estende o Preprocessador Universal para permitir processamento
paralelo de grandes volumes de textos, utilizando Dask para distribuição
de tarefas e otimização de performance.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Callable
from functools import partial

# Configuração de logging
logger = logging.getLogger(__name__)

# Importações condicionais para gerenciar dependências
try:
    import dask
    import dask.bag as db
    from dask.diagnostics import ProgressBar
    DASK_AVAILABLE = True
except ImportError:
    logger.warning("Dask não está disponível. Processamento paralelo será limitado.")
    DASK_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    logger.warning("Ray não está disponível. Fallback para processamento alternativo.")
    RAY_AVAILABLE = False

# Importar a classe base do preprocessador
from src.preprocessor.base import PreprocessorUniversal


class ParallelPreprocessor:
    """
    Wrapper para execução paralela do PreprocessorUniversal.
    
    Esta classe encapsula o PreprocessorUniversal e fornece métodos
    para processamento paralelo de textos utilizando Dask ou Ray.
    """
    
    def __init__(self, preprocessor: PreprocessorUniversal, config: Dict[str, Any] = None):
        """
        Inicializa o processador paralelo.
        
        Args:
            preprocessor: Instância do PreprocessorUniversal
            config: Configurações para processamento paralelo, incluindo:
                   - engine: "dask", "ray", "threads" ou "auto"
                   - n_workers: Número de workers (None = auto)
                   - batch_size: Tamanho do lote para processamento
                   - progress_bar: Exibir barra de progresso
        """
        self.preprocessor = preprocessor
        self.config = config or {}
        
        # Configurações padrão
        self.engine = self.config.get("engine", "auto")
        self.n_workers = self.config.get("n_workers")
        self.batch_size = self.config.get("batch_size", 100)
        self.progress_bar = self.config.get("progress_bar", True)
        
        # Determinar o melhor engine disponível se for "auto"
        if self.engine == "auto":
            if DASK_AVAILABLE:
                self.engine = "dask"
            elif RAY_AVAILABLE:
                self.engine = "ray"
            else:
                self.engine = "threads"
                
        logger.info(f"Inicializando processador paralelo com engine '{self.engine}'")
        
        # Configurar o engine selecionado
        self._setup_engine()
    
    def _setup_engine(self):
        """Configura o engine de processamento paralelo."""
        
        if self.engine == "dask":
            if not DASK_AVAILABLE:
                logger.warning("Dask solicitado, mas não disponível. Usando threads.")
                self.engine = "threads"
            else:
                # Configurar Dask para usar threads ou processos
                dask.config.set(scheduler='processes')  # ou 'threads'
                if self.n_workers:
                    dask.config.set(num_workers=self.n_workers)
                    
        elif self.engine == "ray":
            if not RAY_AVAILABLE:
                logger.warning("Ray solicitado, mas não disponível. Usando threads.")
                self.engine = "threads"
            else:
                # Inicializar Ray
                if not ray.is_initialized():
                    if self.n_workers:
                        ray.init(num_cpus=self.n_workers)
                    else:
                        ray.init()
        
        elif self.engine != "threads":
            logger.warning(f"Engine desconhecido: {self.engine}. Usando threads.")
            self.engine = "threads"
    
    def process_batch(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Processa um lote de textos em paralelo.
        
        Args:
            texts: Lista de textos para processar
            metadatas: Lista de metadados (opcional)
            callback: Função de callback a ser chamada para cada resultado (opcional)
            
        Returns:
            Lista de resultados de processamento
        """
        if not texts:
            logger.warning("Lista de textos vazia para processamento paralelo.")
            return []
            
        # Normalizar metadados
        if not metadatas:
            metadatas = [None] * len(texts)
        elif len(metadatas) != len(texts):
            logger.warning(f"Número de metadados ({len(metadatas)}) diferente do número de textos ({len(texts)})")
            if len(metadatas) < len(texts):
                metadatas = metadatas + [None] * (len(texts) - len(metadatas))
            else:
                metadatas = metadatas[:len(texts)]
        
        # Converter textos vazios para strings vazias
        texts = [t if t is not None else "" for t in texts]
        
        # Escolher a implementação com base no engine
        if self.engine == "dask":
            return self._process_with_dask(texts, metadatas, callback)
        elif self.engine == "ray":
            return self._process_with_ray(texts, metadatas, callback)
        else:
            return self._process_with_threads(texts, metadatas, callback)
    
    def _process_with_dask(self, 
                          texts: List[str], 
                          metadatas: List[Dict[str, Any]],
                          callback: Optional[Callable]) -> List[Dict[str, Any]]:
        """
        Implementação de processamento paralelo com Dask.
        
        Args:
            texts: Lista de textos
            metadatas: Lista de metadados
            callback: Função de callback
            
        Returns:
            Lista de resultados
        """
        # Criar pares de texto e metadados
        pairs = list(zip(texts, metadatas))
        
        # Dividir em lotes para melhor performance
        bag = db.from_sequence(pairs, partition_size=self.batch_size)
        
        # Função de processamento para cada par
        def process_pair(pair):
            text, metadata = pair
            result = self.preprocessor.process(text, metadata)
            if callback:
                callback(result)
            return result
        
        # Aplicar processamento em paralelo
        results_bag = bag.map(process_pair)
        
        # Executar e coletar resultados
        if self.progress_bar:
            with ProgressBar():
                results = results_bag.compute()
        else:
            results = results_bag.compute()
            
        return list(results)
    
    def _process_with_ray(self, 
                         texts: List[str], 
                         metadatas: List[Dict[str, Any]],
                         callback: Optional[Callable]) -> List[Dict[str, Any]]:
        """
        Implementação de processamento paralelo com Ray.
        
        Args:
            texts: Lista de textos
            metadatas: Lista de metadados
            callback: Função de callback
            
        Returns:
            Lista de resultados
        """
        # Definir tarefa remota
        @ray.remote
        def process_text(text, metadata):
            result = self.preprocessor.process(text, metadata)
            return result
        
        # Submeter tarefas para execução paralela
        futures = [process_text.remote(text, metadata) 
                  for text, metadata in zip(texts, metadatas)]
        
        # Obter resultados (com progressão se necessário)
        if self.progress_bar:
            import tqdm
            results = []
            for future in tqdm.tqdm(ray.get(futures), total=len(futures)):
                results.append(future)
                if callback:
                    callback(future)
        else:
            results = ray.get(futures)
            if callback:
                for result in results:
                    callback(result)
                    
        return results
    
    def _process_with_threads(self, 
                             texts: List[str], 
                             metadatas: List[Dict[str, Any]],
                             callback: Optional[Callable]) -> List[Dict[str, Any]]:
        """
        Implementação de processamento com ThreadPoolExecutor.
        
        Args:
            texts: Lista de textos
            metadatas: Lista de metadados
            callback: Função de callback
            
        Returns:
            Lista de resultados
        """
        from concurrent.futures import ThreadPoolExecutor
        import tqdm
        
        results = []
        n_workers = self.n_workers or os.cpu_count()
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Criar tarefas
            futures = [executor.submit(self.preprocessor.process, text, metadata) 
                      for text, metadata in zip(texts, metadatas)]
            
            # Obter resultados
            if self.progress_bar:
                for future in tqdm.tqdm(futures, total=len(futures)):
                    result = future.result()
                    results.append(result)
                    if callback:
                        callback(result)
            else:
                for future in futures:
                    result = future.result()
                    results.append(result)
                    if callback:
                        callback(result)
                        
        return results
    
    def process_stream(self, 
                      texts_stream, 
                      batch_size: Optional[int] = None,
                      callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Processa um stream de textos em lotes paralelos.
        
        Esta função é útil para processar grandes volumes de dados que não cabem
        na memória, processando-os em lotes.
        
        Args:
            texts_stream: Iterador ou gerador que produz tuplas (text, metadata)
            batch_size: Tamanho do lote (se None, usa o valor da configuração)
            callback: Função de callback a ser chamada para cada lote processado
            
        Returns:
            Estatísticas do processamento
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        stats = {
            "total_batches": 0,
            "total_texts": 0,
            "errors": 0,
            "success_rate": 0
        }
        
        batch_texts = []
        batch_metadatas = []
        
        # Processar o stream em lotes
        for item in texts_stream:
            # Desempacotar texto e metadados
            if isinstance(item, tuple) and len(item) == 2:
                text, metadata = item
            else:
                text = item
                metadata = None
                
            batch_texts.append(text)
            batch_metadatas.append(metadata)
            
            # Quando o lote estiver completo, processar
            if len(batch_texts) >= batch_size:
                try:
                    results = self.process_batch(batch_texts, batch_metadatas)
                    stats["total_batches"] += 1
                    stats["total_texts"] += len(batch_texts)
                    
                    # Verificar erros nos resultados
                    for result in results:
                        if "error" in result:
                            stats["errors"] += 1
                    
                    if callback:
                        callback(results)
                        
                except Exception as e:
                    logger.error(f"Erro ao processar lote: {str(e)}")
                    stats["errors"] += len(batch_texts)
                    stats["total_batches"] += 1
                    stats["total_texts"] += len(batch_texts)
                
                # Limpar o lote para o próximo
                batch_texts = []
                batch_metadatas = []
        
        # Processar o último lote parcial, se houver
        if batch_texts:
            try:
                results = self.process_batch(batch_texts, batch_metadatas)
                stats["total_batches"] += 1
                stats["total_texts"] += len(batch_texts)
                
                # Verificar erros nos resultados
                for result in results:
                    if "error" in result:
                        stats["errors"] += 1
                
                if callback:
                    callback(results)
                    
            except Exception as e:
                logger.error(f"Erro ao processar último lote: {str(e)}")
                stats["errors"] += len(batch_texts)
                stats["total_batches"] += 1
                stats["total_texts"] += len(batch_texts)
        
        # Calcular taxa de sucesso
        if stats["total_texts"] > 0:
            stats["success_rate"] = 1.0 - (stats["errors"] / stats["total_texts"])
            
        return stats


# Função de fábrica para criar um preprocessador paralelo
def create_parallel_preprocessor(config: Dict[str, Any]) -> ParallelPreprocessor:
    """
    Cria um preprocessador paralelo com a configuração especificada.
    
    Args:
        config: Configuração completa do sistema
        
    Returns:
        Instância de ParallelPreprocessor
    """
    from src.utils.config import get_config_section
    
    # Extrair configurações específicas
    preprocessor_config = get_config_section(config, "preprocessor")
    parallel_config = get_config_section(config, "parallel_processing")
    
    # Criar o preprocessador base
    preprocessor = PreprocessorUniversal(preprocessor_config)
    
    # Criar e retornar o preprocessador paralelo
    return ParallelPreprocessor(preprocessor, parallel_config)
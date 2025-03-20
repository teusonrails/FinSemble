"""
Testes de benchmark para avaliar a performance e robustez do Preprocessador Universal.

Este módulo contém testes de carga, stress e robustez para garantir que o
preprocessador funcione adequadamente em diferentes cenários, incluindo
volumes grandes de dados e casos de borda.
"""

import os
import sys
import time
import random
import logging
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# Adicionar diretório raiz do projeto ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal
from src.preprocessor.parallel import ParallelPreprocessor, create_parallel_preprocessor


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("benchmark")


class PreprocessorBenchmark:
    """
    Classe para execução de testes de benchmark no Preprocessador Universal.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa o benchmark com a configuração especificada.
        
        Args:
            config_path: Caminho para o arquivo de configuração
        """
        self.config = load_config(config_path)
        self.preprocessor_config = get_config_section(self.config, "preprocessor")
        self.parallel_config = get_config_section(self.config, "parallel_processing")
        
        # Inicializar preprocessadores
        self.sequential_preprocessor = PreprocessorUniversal(self.preprocessor_config)
        self.parallel_preprocessor = create_parallel_preprocessor(self.config)
        
        logger.info("Benchmark inicializado com sucesso")
    
    def generate_random_text(self, min_length: int = 100, max_length: int = 1000) -> str:
        """
        Gera um texto aleatório para testes.
        
        Args:
            min_length: Comprimento mínimo do texto
            max_length: Comprimento máximo do texto
            
        Returns:
            Texto aleatório
        """
        length = random.randint(min_length, max_length)
        
        # Lista de palavras financeiras para tornar o texto mais realista
        financial_words = [
            "lucro", "prejuízo", "receita", "despesa", "ativo", "passivo",
            "investimento", "dividendo", "ação", "mercado", "bolsa", "índice",
            "taxa", "juros", "câmbio", "dólar", "euro", "real", "inflação",
            "valorização", "desvalorização", "trimestre", "balanço", "relatório"
        ]
        
        # Gerar texto com palavras aleatórias e financeiras
        words = []
        for _ in range(length // 5):  # Média de 5 caracteres por palavra
            if random.random() < 0.3:  # 30% de chance de usar palavra financeira
                words.append(random.choice(financial_words))
            else:
                word_length = random.randint(2, 10)
                words.append(''.join(random.choice(string.ascii_lowercase) for _ in range(word_length)))
        
        # Formar sentenças
        sentences = []
        current_sentence = []
        
        for word in words:
            current_sentence.append(word)
            if len(current_sentence) >= random.randint(5, 15) or word == words[-1]:
                sentence = ' '.join(current_sentence)
                sentence = sentence[0].upper() + sentence[1:] + '.'
                sentences.append(sentence)
                current_sentence = []
        
        return ' '.join(sentences)
    
    def generate_test_dataset(self, 
                             num_samples: int = 100, 
                             include_edge_cases: bool = True) -> List[Dict[str, Any]]:
        """
        Gera um conjunto de dados para testes.
        
        Args:
            num_samples: Número de amostras a gerar
            include_edge_cases: Incluir casos de borda
            
        Returns:
            Lista com textos e metadados para teste
        """
        dataset = []
        
        # Gerar amostras normais
        normal_samples = num_samples - 10 if include_edge_cases else num_samples
        for i in range(normal_samples):
            text = self.generate_random_text()
            metadata = {
                "id": i,
                "type": random.choice(["relatorio", "comunicado", "analise"]),
                "date": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
            }
            dataset.append((text, metadata))
        
        # Adicionar casos de borda, se solicitado
        if include_edge_cases:
            # Texto vazio
            dataset.append(("", {"id": normal_samples, "type": "empty"}))
            
            # Texto muito curto
            dataset.append(("a", {"id": normal_samples + 1, "type": "short"}))
            
            # Texto muito longo
            long_text = self.generate_random_text(50000, 100000)
            dataset.append((long_text, {"id": normal_samples + 2, "type": "long"}))
            
            # Texto com caracteres especiais
            special_chars = "áéíóúàèìòùâêîôûãõñçÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÑÇ!@#$%^&*()_+[]{}|;:,.<>?/\\"
            special_text = self.generate_random_text(500, 1000) + special_chars
            dataset.append((special_text, {"id": normal_samples + 3, "type": "special"}))
            
            # Texto com repetições
            repeat_text = "repetição " * 100
            dataset.append((repeat_text, {"id": normal_samples + 4, "type": "repetition"}))
            
            # Metadata None
            dataset.append((self.generate_random_text(), None))
            
            # Metadata inválido
            dataset.append((self.generate_random_text(), "invalid metadata"))
            
            # Texto None
            dataset.append((None, {"id": normal_samples + 7, "type": "none"}))
            
            # Texto não string
            dataset.append((123, {"id": normal_samples + 8, "type": "non_string"}))
            
            # Caracteres de controle
            control_text = self.generate_random_text() + "\x00\x01\x02\x03\x04"
            dataset.append((control_text, {"id": normal_samples + 9, "type": "control"}))
        
        return dataset
    
    def run_sequential_benchmark(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executa benchmark no preprocessador sequencial.
        
        Args:
            dataset: Conjunto de dados para teste
            
        Returns:
            Resultados do benchmark
        """
        logger.info(f"Iniciando benchmark sequencial com {len(dataset)} amostras")
        
        start_time = time.time()
        results = []
        errors = 0
        
        for i, (text, metadata) in enumerate(dataset):
            try:
                result = self.sequential_preprocessor.process(text, metadata)
                results.append(result)
                if "error" in result:
                    errors += 1
            except Exception as e:
                logger.error(f"Erro no processamento sequencial, amostra {i}: {str(e)}")
                errors += 1
        
        total_time = time.time() - start_time
        
        # Compilar estatísticas
        stats = {
            "mode": "sequential",
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "errors": errors,
            "success_rate": (len(results) - errors) / len(dataset) if dataset else 0,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(dataset) if dataset else 0,
            "samples_per_second": len(dataset) / total_time if total_time > 0 else 0,
        }
        
        logger.info(f"Benchmark sequencial concluído: {stats['samples_per_second']:.2f} amostras/segundo")
        return stats
    
    def run_parallel_benchmark(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executa benchmark no preprocessador paralelo.
        
        Args:
            dataset: Conjunto de dados para teste
            
        Returns:
            Resultados do benchmark
        """
        logger.info(f"Iniciando benchmark paralelo com {len(dataset)} amostras")
        
        texts, metadatas = zip(*dataset) if dataset else ([], [])
        
        start_time = time.time()
        try:
            results = self.parallel_preprocessor.process_batch(list(texts), list(metadatas))
            errors = sum(1 for result in results if "error" in result)
        except Exception as e:
            logger.error(f"Erro no processamento paralelo: {str(e)}")
            results = []
            errors = len(dataset)
        
        total_time = time.time() - start_time
        
        # Compilar estatísticas
        stats = {
            "mode": "parallel",
            "engine": self.parallel_preprocessor.engine,
            "total_samples": len(dataset),
            "processed_samples": len(results),
            "errors": errors,
            "success_rate": (len(results) - errors) / len(dataset) if dataset else 0,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / len(dataset) if dataset else 0,
            "samples_per_second": len(dataset) / total_time if total_time > 0 else 0,
        }
        
        logger.info(f"Benchmark paralelo concluído: {stats['samples_per_second']:.2f} amostras/segundo")
        return stats
    
    def run_edge_case_test(self) -> Dict[str, Any]:
        """
        Testa a robustez do preprocessador com casos de borda.
        
        Returns:
            Resultados dos testes
        """
        logger.info("Iniciando testes de casos de borda")
        
        edge_cases = [
            ("", "Texto vazio"),
            (None, "None"),
            (123, "Valor numérico"),
            ("a" * 1000000, "Texto muito longo"),
            ("áéíóúàèìòùâêîôûãõñç", "Acentuação"),
            ("!@#$%^&*()_+", "Símbolos"),
            ("\x00\x01\x02\x03", "Caracteres de controle"),
            ("<script>alert('XSS')</script>", "Código malicioso"),
            ("DROP TABLE users;", "Injeção SQL")
        ]
        
        results = []
        for text, description in edge_cases:
            try:
                result = self.sequential_preprocessor.process(text)
                success = "error" not in result
                results.append({
                    "description": description,
                    "success": success,
                    "error": result.get("error", None),
                    "has_fallback": "warnings" in result,
                    "warnings": result.get("warnings", [])
                })
            except Exception as e:
                results.append({
                    "description": description,
                    "success": False,
                    "error": str(e),
                    "has_fallback": False,
                    "warnings": []
                })
        
        # Compilar estatísticas
        stats = {
            "total_cases": len(edge_cases),
            "successful_cases": sum(1 for r in results if r["success"]),
            "failed_cases": sum(1 for r in results if not r["success"]),
            "cases_with_fallback": sum(1 for r in results if r["has_fallback"]),
            "detailed_results": results
        }
        
        logger.info(f"Testes de casos de borda concluídos: {stats['successful_cases']}/{stats['total_cases']} bem-sucedidos")
        return stats
    
    def run_stress_test(self, 
                       num_samples: int = 1000, 
                       max_concurrent: int = 10) -> Dict[str, Any]:
        """
        Executa teste de stress no preprocessador.
        
        Args:
            num_samples: Número total de amostras
            max_concurrent: Número máximo de processos concorrentes
            
        Returns:
            Resultados do teste
        """
        logger.info(f"Iniciando teste de stress com {num_samples} amostras e {max_concurrent} processos concorrentes")
        
        # Gerar dados para o teste
        dataset = self.generate_test_dataset(num_samples, include_edge_cases=False)
        texts, metadatas = zip(*dataset)
        
        start_time = time.time()
        errors = 0
        processed = 0
        
        # Função para processar um lote
        def process_batch(batch_texts, batch_metadatas):
            nonlocal errors, processed
            try:
                results = self.parallel_preprocessor.process_batch(batch_texts, batch_metadatas)
                batch_errors = sum(1 for result in results if "error" in result)
                errors += batch_errors
                processed += len(results)
                return len(results) - batch_errors
            except Exception as e:
                logger.error(f"Erro no processamento de lote: {str(e)}")
                errors += len(batch_texts)
                return 0
        
        # Dividir em lotes
        batch_size = min(100, num_samples // max_concurrent)
        if batch_size == 0:
            batch_size = 1
            
        batches_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        batches_metadatas = [metadatas[i:i + batch_size] for i in range(0, len(metadatas), batch_size)]
        
        # Executar lotes em paralelo
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(process_batch, batch_texts, batch_metadatas)
                for batch_texts, batch_metadatas in zip(batches_texts, batches_metadatas)
            ]
            
            # Aguardar conclusão
            for future in futures:
                future.result()
        
        total_time = time.time() - start_time
        
        # Compilar estatísticas
        stats = {
            "mode": "stress_test",
            "total_samples": num_samples,
            "concurrent_processes": max_concurrent,
            "batch_size": batch_size,
            "processed_samples": processed,
            "errors": errors,
            "success_rate": (processed - errors) / num_samples if num_samples > 0 else 0,
            "total_time_seconds": total_time,
            "avg_time_per_sample": total_time / num_samples if num_samples > 0 else 0,
            "samples_per_second": num_samples / total_time if total_time > 0 else 0,
        }
        
        logger.info(f"Teste de stress concluído: {stats['samples_per_second']:.2f} amostras/segundo")
        return stats
    
    def run_full_benchmark(self, 
                          small_batch: int = 100, 
                          medium_batch: int = 500, 
                          large_batch: int = 1000) -> Dict[str, Any]:
        """
        Executa uma bateria completa de testes de benchmark.
        
        Args:
            small_batch: Tamanho do lote pequeno
            medium_batch: Tamanho do lote médio
            large_batch: Tamanho do lote grande
            
        Returns:
            Resultados completos dos testes
        """
        logger.info("Iniciando bateria completa de testes de benchmark")
        
        results = {
            "sequential": {},
            "parallel": {},
            "edge_cases": {},
            "stress_test": {}
        }
        
        # Testes sequenciais
        for size, name in [(small_batch, "small"), (medium_batch, "medium")]:
            dataset = self.generate_test_dataset(size)
            results["sequential"][name] = self.run_sequential_benchmark(dataset)
        
        # Testes paralelos
        for size, name in [(small_batch, "small"), (medium_batch, "medium"), (large_batch, "large")]:
            dataset = self.generate_test_dataset(size)
            results["parallel"][name] = self.run_parallel_benchmark(dataset)
        
        # Teste de casos de borda
        results["edge_cases"] = self.run_edge_case_test()
        
        # Teste de stress
        results["stress_test"] = self.run_stress_test(large_batch)
        
        # Resumo comparativo
        if "small" in results["sequential"] and "small" in results["parallel"]:
            speedup = (
                results["parallel"]["small"]["samples_per_second"] / 
                results["sequential"]["small"]["samples_per_second"]
                if results["sequential"]["small"]["samples_per_second"] > 0 else 0
            )
            
            results["summary"] = {
                "sequential_throughput": results["sequential"]["small"]["samples_per_second"],
                "parallel_throughput": results["parallel"]["small"]["samples_per_second"],
                "speedup_factor": speedup,
                "edge_case_success_rate": results["edge_cases"]["successful_cases"] / results["edge_cases"]["total_cases"],
                "stress_test_throughput": results["stress_test"]["samples_per_second"]
            }
            
            logger.info(f"Speedup paralelo vs. sequencial: {speedup:.2f}x")
        
        logger.info("Bateria completa de testes concluída")
        return results
    
    def save_benchmark_results(self, results: Dict[str, Any], filepath: str = "benchmark_results.json"):
        """
        Salva os resultados do benchmark em um arquivo JSON.
        
        Args:
            results: Resultados do benchmark
            filepath: Caminho para o arquivo de saída
        """
        import json
        
        # Converter tipos não serializáveis
        def json_serializable(obj):
            if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(filepath, 'w') as f:
            json.dump(results, f, default=json_serializable, indent=2)
            
        logger.info(f"Resultados salvos em {filepath}")


if __name__ == "__main__":
    # Executar benchmark completo
    benchmark = PreprocessorBenchmark()
    results = benchmark.run_full_benchmark(
        small_batch=50,   # Para execução rápida, ajuste conforme necessário
        medium_batch=100,
        large_batch=200
    )
    
    # Salvar resultados
    os.makedirs("tests/benchmarks/results", exist_ok=True)
    benchmark.save_benchmark_results(results, "tests/benchmarks/results/benchmark_results.json")
    
    # Exibir resumo
    if "summary" in results:
        print("\nResumo do Benchmark:")
        print(f"Throughput Sequencial: {results['summary']['sequential_throughput']:.2f} amostras/segundo")
        print(f"Throughput Paralelo: {results['summary']['parallel_throughput']:.2f} amostras/segundo")
        print(f"Speedup: {results['summary']['speedup_factor']:.2f}x")
        print(f"Taxa de Sucesso em Casos de Borda: {results['summary']['edge_case_success_rate'] * 100:.1f}%")
        print(f"Throughput no Teste de Stress: {results['summary']['stress_test_throughput']:.2f} amostras/segundo")
"""
Script de demonstração do Preprocessador Universal aprimorado do FinSemble.

Este script demonstra as melhorias implementadas no preprocessador,
incluindo validação de inputs, gestão de recursos externos,
processamento paralelo e tratamento de casos de borda.
"""

import os
import sys
import json
import time
import logging
from pprint import pprint

# Adicionar o diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.config import load_config, get_config_section
from src.utils.validators import sanitize_text, sanitize_metadata
from src.utils.resource_manager import ResourceManager, check_environment
from src.preprocessor.base import PreprocessorUniversal
from src.preprocessor.parallel import ParallelPreprocessor, create_parallel_preprocessor


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("demo_preprocessador")


def print_section(title):
    """Imprime um título de seção formatado."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, '='))
    print("=" * 80 + "\n")


def print_dict(data, prefix='', max_depth=3, current_depth=0):
    """
    Imprime um dicionário de forma legível, com controle de profundidade.
    
    Args:
        data: Dicionário ou objeto a imprimir
        prefix: Prefixo para indentação
        max_depth: Profundidade máxima a mostrar
        current_depth: Profundidade atual (para recursão)
    """
    if current_depth >= max_depth:
        print(f"{prefix}...")
        return
        
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}{key}:")
                print_dict(value, prefix + '  ', max_depth, current_depth + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                print(f"{prefix}{key}: [")
                for i, item in enumerate(value[:3]):  # Limitar a 3 itens
                    print(f"{prefix}  Item {i}:")
                    print_dict(item, prefix + '    ', max_depth, current_depth + 1)
                if len(value) > 3:
                    print(f"{prefix}  ... e mais {len(value) - 3} itens")
                print(f"{prefix}]")
            elif isinstance(value, list) and len(value) > 10:
                print(f"{prefix}{key}: [{len(value)} items]")
                # Mostrar alguns elementos do início e do fim
                sample = value[:3] + ["..."] + value[-3:] if len(value) > 6 else value
                print(f"{prefix}  {sample}")
            else:
                str_value = str(value)
                if len(str_value) > 100:
                    str_value = str_value[:97] + "..."
                print(f"{prefix}{key}: {str_value}")
    else:
        print(f"{prefix}{data}")


def demo_verificacao_recursos():
    """Demonstra a verificação de recursos externos."""
    print_section("Verificação de Recursos Externos")
    
    print("Verificando disponibilidade de recursos necessários...\n")
    status = check_environment()
    
    print(f"Status geral: {status['status']}")
    print(f"NLTK: {'OK' if status['nltk'].get('status', False) else 'FALHA'}")
    print(f"spaCy: {'OK' if status['spacy'].get('status', False) else 'FALHA'}")
    
    if status["spacy"].get("best_model"):
        print(f"Melhor modelo spaCy disponível: {status['spacy']['best_model']}")
    
    print("\nDetalhes das verificações:")
    print_dict(status)


def demo_validacao_inputs():
    """Demonstra a validação e sanitização de inputs."""
    print_section("Validação e Sanitização de Inputs")
    
    # Exemplos de textos com problemas
    exemplos = [
        ("Texto normal sem problemas", "Texto normal"),
        ("", "Texto vazio"),
        (None, "None"),
        (123, "Valor numérico"),
        ("Texto com \x00\x01 caracteres de controle", "Caracteres de controle"),
        ("<script>alert('XSS')</script>", "Código potencialmente malicioso"),
        ("áéíóúàèìòùâêîôûãõñç", "Caracteres acentuados")
    ]
    
    # Exemplos de metadados com problemas
    metadados = [
        ({"source": "exemplo", "date": "2023-05-15"}, "Metadados válidos"),
        (None, "Metadados None"),
        ("não é um dicionário", "Metadados não-dicionário"),
        ({"data": "x" * 1000}, "Metadados muito extensos"),
        ({"campo_inválido": "<script>alert('XSS')</script>"}, "Metadados com código")
    ]
    
    print("Validação e sanitização de textos:")
    for texto, descricao in exemplos:
        sanitizado = sanitize_text(texto)
        print(f"\n{descricao}:")
        print(f"  Original: {texto}")
        print(f"  Sanitizado: {sanitizado}")
    
    print("\nValidação e sanitização de metadados:")
    for metadata, descricao in metadados:
        sanitizado = sanitize_metadata(metadata)
        print(f"\n{descricao}:")
        print(f"  Original: {metadata}")
        print(f"  Sanitizado: {sanitizado}")


def demo_processamento_sequencial():
    """Demonstra o processamento sequencial com tratamento de casos de borda."""
    print_section("Processamento Sequencial com Tratamento de Casos de Borda")
    
    config = load_config()
    preprocessor_config = get_config_section(config, "preprocessor")
    
    # Ativar fallbacks e monitoramento de performance
    preprocessor_config["fallback_enabled"] = True
    preprocessor_config["performance_monitoring"] = True
    
    preprocessor = PreprocessorUniversal(preprocessor_config)
    
    # Exemplos de textos para processamento
    exemplos = [
        # Caso normal
        ("""
        A Empresa XYZ S.A. comunica aos seus acionistas e ao mercado em geral que registrou
        um lucro líquido de R$ 850 milhões no primeiro trimestre de 2023, o que representa
        um crescimento de 15% em relação ao mesmo período do ano anterior.
        """, {"source": "comunicado", "date": "2023-05-15"}, "Caso normal"),
        
        # Texto vazio (caso de borda)
        ("", {"source": "vazio"}, "Texto vazio"),
        
        # Texto muito curto (caso de borda)
        ("OK", {"source": "curto"}, "Texto muito curto"),
        
        # Texto com caracteres especiais (caso de borda)
        ("Situação econômica difícil! Taxa de câmbio R$/US$ = 4.95 #análise @mercado",
         {"source": "especial"}, "Caracteres especiais"),
         
        # Texto muito longo (truncado na saída)
        ("A" * 10000, {"source": "longo"}, "Texto muito longo")
    ]
    
    for texto, metadata, descricao in exemplos:
        print(f"\nProcessando: {descricao}")
        start_time = time.time()
        
        resultado = preprocessor.process(texto, metadata)
        
        processing_time = time.time() - start_time
        print(f"Tempo de processamento: {processing_time*1000:.2f}ms")
        
        if "error" in resultado:
            print(f"Erro: {resultado['error']}")
        
        if "warnings" in resultado:
            print(f"Avisos: {resultado['warnings']}")
        
        # Mostrar um resumo dos resultados
        print("\nResumo dos resultados:")
        if "normalized_text" in resultado:
            texto_norm = resultado["normalized_text"]
            print(f"Texto normalizado: {texto_norm[:100]}..." if len(texto_norm) > 100 else texto_norm)
        
        if "features" in resultado:
            print("\nCaracterísticas extraídas:")
            for tipo, features in resultado["features"].items():
                print(f"  {tipo}: {len(features)} características")
            
            # Mostrar algumas características de exemplo
            if "type_features" in resultado["features"]:
                print("\nExemplo de características de tipo:")
                print_dict(resultado["features"]["type_features"], "  ")
            
            if "sentiment_features" in resultado["features"]:
                print("\nExemplo de características de sentimento:")
                print_dict(resultado["features"]["sentiment_features"], "  ")
        
        if "derived_metadata" in resultado:
            print("\nMetadados derivados:")
            print_dict(resultado["derived_metadata"], "  ")
        
        if "processing_stats" in resultado:
            print("\nEstatísticas de processamento:")
            print_dict(resultado["processing_stats"], "  ")


def demo_processamento_paralelo():
    """Demonstra o processamento paralelo de lotes de textos."""
    print_section("Processamento Paralelo de Lotes")
    
    config = load_config()
    
    # Criar o preprocessador paralelo
    parallel_preprocessor = create_parallel_preprocessor(config)
    
    print(f"Preprocessador paralelo inicializado com engine: {parallel_preprocessor.engine}")
    
    # Gerar um lote de textos para processamento
    textos = [
        f"Texto de exemplo {i} para processamento em lote. Este é um texto financeiro de teste."
        for i in range(20)
    ]
    
    # Adicionar alguns casos de borda
    textos.append("")  # Texto vazio
    textos.append("A" * 5000)  # Texto longo
    textos.append(None)  # None
    textos.append(123)  # Não-string
    
    metadados = [{"id": i, "batch": "demo"} for i in range(len(textos))]
    
    print(f"Processando lote de {len(textos)} textos em paralelo...")
    
    start_time = time.time()
    resultados = parallel_preprocessor.process_batch(textos, metadados)
    total_time = time.time() - start_time
    
    print(f"Processamento concluído em {total_time:.2f} segundos")
    print(f"Tempo médio por texto: {(total_time/len(textos))*1000:.2f}ms")
    print(f"Textos processados por segundo: {len(textos)/total_time:.2f}")
    
    # Verificar erros
    erros = sum(1 for r in resultados if "error" in r)
    print(f"Textos com erro: {erros}/{len(textos)}")
    
    # Verificar avisos
    avisos = sum(1 for r in resultados if "warnings" in r)
    print(f"Textos com avisos: {avisos}/{len(textos)}")
    
    # Mostrar exemplos de resultados
    print("\nExemplos de resultados:")
    for i in [0, len(textos)//2, len(textos)-2]:  # Início, meio e fim
        if i < len(resultados):
            print(f"\nResultado {i}:")
            resultado = resultados[i]
            
            if "error" in resultado:
                print(f"Erro: {resultado['error']}")
            elif "warnings" in resultado:
                print(f"Avisos: {resultado['warnings']}")
                
            if "normalized_text" in resultado:
                texto_norm = resultado["normalized_text"]
                print(f"Texto normalizado: {texto_norm[:50]}..." if len(texto_norm) > 50 else texto_norm)
                
            if "features" in resultado:
                print("Características extraídas:")
                for tipo, features in resultado["features"].items():
                    print(f"  {tipo}: {len(features)} características")
    
    # Salvar os resultados
    output_path = "data/processed/batch_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Simplificar os resultados para salvar
    simple_results = []
    for r in resultados:
        simple_result = {
            "has_error": "error" in r,
            "has_warnings": "warnings" in r,
            "num_features": sum(len(features) for features in r.get("features", {}).values()),
            "metadata": r.get("metadata", {}),
            "derived_metadata": r.get("derived_metadata", {})
        }
        simple_results.append(simple_result)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simple_results, f, ensure_ascii=False, indent=2)
        
    print(f"\nResultados simplificados salvos em '{output_path}'")


def demo_processamento_streaming():
    """Demonstra o processamento de um stream de textos."""
    print_section("Processamento de Stream de Textos")
    
    config = load_config()
    
    # Criar o preprocessador paralelo
    parallel_preprocessor = create_parallel_preprocessor(config)
    
    print(f"Preprocessador paralelo inicializado com engine: {parallel_preprocessor.engine}")
    
    # Simular um stream de textos (na prática seria um arquivo grande ou API)
    def text_stream(n=50):
        """Gera um stream de textos de exemplo."""
        for i in range(n):
            text = f"Texto de stream {i}. Este é um exemplo para processamento de stream."
            metadata = {"id": i, "stream": "demo"}
            yield (text, metadata)
    
    # Função de callback para cada lote processado
    def batch_callback(results):
        """Callback chamado após o processamento de cada lote."""
        successful = sum(1 for r in results if "error" not in r)
        print(f"Lote processado: {successful}/{len(results)} textos bem-sucedidos")
    
    print("Iniciando processamento de stream...")
    start_time = time.time()
    
    # Processar o stream
    stats = parallel_preprocessor.process_stream(
        text_stream(50),  # 50 textos no stream
        batch_size=10,    # Processar em lotes de 10
        callback=batch_callback
    )
    
    total_time = time.time() - start_time
    
    print("\nProcessamento de stream concluído!")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"Textos processados: {stats['total_texts']}")
    print(f"Lotes processados: {stats['total_batches']}")
    print(f"Erros: {stats['errors']}")
    print(f"Taxa de sucesso: {stats['success_rate']*100:.1f}%")
    print(f"Textos por segundo: {stats['total_texts']/total_time:.2f}")


def demo_preprocessador():
    """Função principal que executa todas as demonstrações."""
    print_section("Demonstração do Preprocessador Universal Aprimorado")
    
    # 1. Verificação de recursos
    demo_verificacao_recursos()
    
    # 2. Validação de inputs
    demo_validacao_inputs()
    
    # 3. Processamento sequencial
    demo_processamento_sequencial()
    
    # 4. Processamento paralelo em lote
    demo_processamento_paralelo()
    
    # 5. Processamento de stream
    demo_processamento_streaming()
    
    print_section("Demonstração Concluída")
    print("O Preprocessador Universal Aprimorado foi demonstrado com sucesso!")


if __name__ == "__main__":
    demo_preprocessador()
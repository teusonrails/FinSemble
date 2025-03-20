import sys
import os
import time
from pprint import pprint

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config
from src.preprocessor.parallel import create_parallel_preprocessor

# Carregar configuração
config = load_config()

# Criar preprocessador paralelo
parallel_preprocessor = create_parallel_preprocessor(config)

# Gerar textos de teste
def generate_texts(n=20):
    texts = []
    for i in range(n):
        texts.append(f"""
        Comunicado ao Mercado - {i+1}
        A Empresa ABC S.A. informa aos seus acionistas e ao mercado em geral que
        adquiriu a empresa XYZ Ltda. por R$ {(i+1)*50} milhões. A aquisição faz parte
        da estratégia de expansão da companhia e deve contribuir para um crescimento
        de {(i+1)*2}% no faturamento nos próximos 12 meses.
        """)
    return texts

# Gerar textos
textos = generate_texts(20)
metadados = [{"id": i, "source": "test"} for i in range(len(textos))]

# Processar em lote
print(f"Processando lote de {len(textos)} textos em paralelo...")
start_time = time.time()
resultados = parallel_preprocessor.process_batch(textos, metadados)
total_time = time.time() - start_time

print(f"Processamento concluído em {total_time:.2f} segundos")
print(f"Tempo médio por texto: {(total_time/len(textos))*1000:.2f}ms")
print(f"Textos processados por segundo: {len(textos)/total_time:.2f}")

# Verificar resultados
erros = sum(1 for r in resultados if "error" in r)
avisos = sum(1 for r in resultados if "warnings" in r)
print(f"Textos com erro: {erros}/{len(textos)}")
print(f"Textos com avisos: {avisos}/{len(textos)}")

# Mostrar exemplo de resultado
print("\nExemplo de resultado (primeiro texto):")
resultado = resultados[0]
if "normalized_text" in resultado:
    print("\nTexto normalizado:")
    print(resultado["normalized_text"][:200] + "...")

if "features" in resultado:
    print("\nCaracterísticas extraídas (contagem):")
    for tipo, features in resultado["features"].items():
        print(f"  {tipo}: {len(features)} características")
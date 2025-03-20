import sys
import os
from pprint import pprint

# Adicionar diretório raiz ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))

from src.utils.config import load_config, get_config_section
from src.preprocessor.base import PreprocessorUniversal

# Carregar configuração
config = load_config()
preprocessor_config = get_config_section(config, "preprocessor")

# Inicializar preprocessador
preprocessor = PreprocessorUniversal(preprocessor_config)

# Texto de teste
texto = """
A Empresa XYZ S.A. (XYZS3) comunica aos seus acionistas e ao mercado em geral que
registrou um crescimento significativo de 25% no lucro líquido do segundo trimestre de 2023,
alcançando R$ 450 milhões. Este resultado positivo foi impulsionado pela expansão das operações
internacionais e pelo aumento da eficiência operacional, que elevou a margem EBITDA para 32%.
"""

# Processar texto
resultado = preprocessor.process(texto)

# Exibir resultados
print("\nTexto Original:")
print(texto)

print("\nTexto Normalizado:")
print(resultado["normalized_text"])

print("\nTokens (primeiros 10):")
if "tokens" in resultado and "words" in resultado["tokens"]:
    print(resultado["tokens"]["words"][:10])

print("\nCaracterísticas Extraídas:")
if "features" in resultado:
    for tipo, features in resultado["features"].items():
        print(f"\n{tipo}: {len(features)} características")
        # Mostrar algumas características
        for i, (k, v) in enumerate(features.items()):
            if i >= 5:  # Limitar a 5 características por tipo
                break
            print(f"  {k}: {v}")

print("\nMetadados Derivados:")
if "derived_metadata" in resultado:
    pprint(resultado["derived_metadata"])

print("\nEstatísticas de Processamento:")
if "processing_stats" in resultado:
    pprint(resultado["processing_stats"])
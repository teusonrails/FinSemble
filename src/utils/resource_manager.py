"""
Módulo para gerenciamento de recursos externos do FinSemble.

Este módulo fornece funcionalidades para verificar, baixar e gerenciar
recursos externos como modelos do spaCy, recursos do NLTK e outros
componentes necessários para o funcionamento do sistema.
"""

import os
import sys
import logging
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Set

# Configuração de logging
logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Gerencia recursos externos necessários para o FinSemble.
    
    Esta classe centraliza a verificação e o download de recursos como
    modelos de linguagem, léxicos e outros componentes, garantindo que
    estejam disponíveis antes da execução dos componentes do sistema.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o gerenciador de recursos.
        
        Args:
            config: Configurações para o gerenciador de recursos, incluindo
                   caminhos, versões de modelos e preferências de fallback.
        """
        self.config = config
        self.resources_path = config.get("resources_path", "data/resources")
        self.language = config.get("language", "portuguese")
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.offline_mode = config.get("offline_mode", False)
        
        # Garantir que o diretório de recursos exista
        os.makedirs(self.resources_path, exist_ok=True)
        
        # Mapear abreviações de idiomas para códigos
        self.language_map = {
            "portuguese": "pt",
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "italian": "it",
        }
        
        # Lista de recursos básicos NLTK necessários
        self.nltk_resources = {
            "tokenizers/punkt": "Tokenizador de sentenças",
            "corpora/stopwords": "Lista de stopwords",
            "corpora/wordnet": "WordNet para lematização",
        }
        
        # Modelos do spaCy por idioma
        self.spacy_models = {
            "pt": ["pt_core_news_sm", "pt_core_news_lg"],
            "en": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
            "es": ["es_core_news_sm", "es_core_news_md"],
        }
        
        # Mapeamento de fallback para modelos spaCy
        self.spacy_fallbacks = {
            "pt_core_news_lg": "pt_core_news_sm",
            "pt_core_news_sm": "en_core_web_sm",
            "en_core_web_lg": "en_core_web_md",
            "en_core_web_md": "en_core_web_sm",
        }
    
    def check_internet_connection(self) -> bool:
        """
        Verifica se há conexão com a internet.
        
        Returns:
            True se houver conexão, False caso contrário.
        """
        if self.offline_mode:
            logger.info("Modo offline ativado. Assumindo sem conexão.")
            return False
            
        import socket
        try:
            # Tenta conectar ao Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
    
    def verify_nltk_resources(self) -> Dict[str, bool]:
        """
        Verifica se os recursos necessários do NLTK estão disponíveis.
        
        Returns:
            Dicionário com o status de cada recurso.
        """
        import nltk
        
        resource_status = {}
        has_connection = self.check_internet_connection()
        
        for resource_path, description in self.nltk_resources.items():
            try:
                nltk.data.find(resource_path)
                resource_status[resource_path] = True
                logger.info(f"Recurso NLTK disponível: {description}")
            except LookupError:
                logger.warning(f"Recurso NLTK não encontrado: {description}")
                resource_status[resource_path] = False
                
                # Tentar download se houver conexão
                if has_connection:
                    try:
                        logger.info(f"Tentando baixar recurso NLTK: {description}")
                        nltk.download(resource_path.split('/')[-1])
                        resource_status[resource_path] = True
                    except Exception as e:
                        logger.error(f"Erro ao baixar recurso NLTK: {str(e)}")
        
        return resource_status
    
    def verify_spacy_models(self) -> Dict[str, bool]:
        """
        Verifica se os modelos necessários do spaCy estão disponíveis.
        
        Returns:
            Dicionário com o status de cada modelo.
        """
        import importlib.util
        import spacy
        
        lang_code = self.language_map.get(self.language, "en")
        models_to_check = self.spacy_models.get(lang_code, ["en_core_web_sm"])
        
        model_status = {}
        has_connection = self.check_internet_connection()
        
        for model_name in models_to_check:
            try:
                # Verifica se o modelo já está carregado ou pode ser carregado
                if model_name in spacy.util.get_installed_models():
                    model_status[model_name] = True
                    logger.info(f"Modelo spaCy disponível: {model_name}")
                    continue
                    
                # Verifica se o módulo Python correspondente ao modelo está instalado
                model_spec = importlib.util.find_spec(model_name)
                if model_spec is not None:
                    model_status[model_name] = True
                    logger.info(f"Modelo spaCy disponível: {model_name}")
                else:
                    raise ImportError(f"Modelo {model_name} não encontrado")
                    
            except Exception as e:
                logger.warning(f"Modelo spaCy não disponível: {model_name} - {str(e)}")
                model_status[model_name] = False
                
                # Tentar download se houver conexão
                if has_connection:
                    try:
                        logger.info(f"Tentando baixar modelo spaCy: {model_name}")
                        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
                        model_status[model_name] = True
                    except Exception as e:
                        logger.error(f"Erro ao baixar modelo spaCy: {str(e)}")
        
        return model_status
    
    def get_best_available_spacy_model(self) -> str:
        """
        Retorna o melhor modelo spaCy disponível para o idioma configurado.
        
        Returns:
            Nome do modelo spaCy disponível, ou None se nenhum estiver disponível.
        """
        import spacy
        
        lang_code = self.language_map.get(self.language, "en")
        models_to_check = self.spacy_models.get(lang_code, ["en_core_web_sm"])
        
        # Primeiro tenta os modelos na ordem de preferência
        for model_name in models_to_check:
            if model_name in spacy.util.get_installed_models():
                logger.info(f"Usando modelo spaCy: {model_name}")
                return model_name
        
        # Se não encontrar, tenta fallbacks
        if self.fallback_enabled:
            for model_name in models_to_check:
                fallback = self.spacy_fallbacks.get(model_name)
                if fallback and fallback in spacy.util.get_installed_models():
                    logger.warning(f"Usando modelo spaCy de fallback: {fallback} (em vez de {model_name})")
                    return fallback
        
        # Último recurso: modelo em inglês básico
        if "en_core_web_sm" in spacy.util.get_installed_models():
            logger.warning("Usando modelo spaCy em inglês como último recurso")
            return "en_core_web_sm"
            
        logger.error("Nenhum modelo spaCy disponível")
        return None
    
    def initialize_resources(self) -> Dict[str, Any]:
        """
        Inicializa todos os recursos externos necessários.
        
        Returns:
            Dicionário com os recursos inicializados e seu status.
        """
        resources = {
            "status": "success",
            "nltk": {},
            "spacy": {},
            "lexicons": {},
        }
        
        # Verificar recursos NLTK
        try:
            nltk_status = self.verify_nltk_resources()
            resources["nltk"] = {
                "status": all(nltk_status.values()),
                "resources": nltk_status
            }
        except Exception as e:
            logger.error(f"Erro ao verificar recursos NLTK: {str(e)}")
            resources["nltk"] = {"status": False, "error": str(e)}
            resources["status"] = "partial"
        
        # Verificar modelos spaCy
        try:
            spacy_status = self.verify_spacy_models()
            best_model = self.get_best_available_spacy_model()
            resources["spacy"] = {
                "status": any(spacy_status.values()),
                "models": spacy_status,
                "best_model": best_model
            }
        except Exception as e:
            logger.error(f"Erro ao verificar modelos spaCy: {str(e)}")
            resources["spacy"] = {"status": False, "error": str(e)}
            resources["status"] = "partial"
        
        # Verificar léxicos financeiros (implementação futura)
        resources["lexicons"] = {"status": True, "message": "Léxicos financeiros não implementados ainda"}
        
        # Atualizar status geral
        if not resources["nltk"].get("status", False) and not resources["spacy"].get("status", False):
            resources["status"] = "failure"
            logger.error("Falha na inicialização de recursos críticos")
        elif resources["status"] != "partial":
            logger.info("Todos os recursos inicializados com sucesso")
        
        return resources


# Função auxiliar para verificação rápida do ambiente
def check_environment(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Verifica rapidamente se o ambiente tem os recursos necessários.
    
    Args:
        config: Configurações para o gerenciador de recursos (opcional)
        
    Returns:
        Dicionário com o status dos recursos
    """
    if config is None:
        config = {}
    
    resource_manager = ResourceManager(config)
    return resource_manager.initialize_resources()


if __name__ == "__main__":
    # Configuração de logging para execução direta
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verificar ambiente quando executado diretamente
    print("Verificando recursos do ambiente...")
    status = check_environment()
    
    print("\nStatus dos recursos:")
    print(f"Status geral: {status['status']}")
    print(f"NLTK: {'OK' if status['nltk'].get('status', False) else 'FALHA'}")
    print(f"spaCy: {'OK' if status['spacy'].get('status', False) else 'FALHA'}")
    
    if status["spacy"].get("best_model"):
        print(f"Melhor modelo spaCy disponível: {status['spacy']['best_model']}")
    
    print("\nDetalhes:")
    for resource_type, details in status.items():
        if resource_type != "status":
            print(f"\n{resource_type.upper()}:")
            for key, value in details.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                else:
                    print(f"  {key}: {value}")
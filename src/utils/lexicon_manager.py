"""
Módulo para gerenciamento de léxicos de sentimento do sistema FinSemble.

Este módulo fornece funcionalidades para carregar, manter e atualizar léxicos
de sentimento utilizados na análise de textos financeiros.
"""

import os
import json
import csv
import logging
from typing import Dict, List, Any, Optional

# Configuração de logging
logger = logging.getLogger(__name__)

class LexiconManager:
    """Gerenciador de léxicos de sentimento."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o gerenciador de léxicos.
        
        Args:
            config: Configurações incluindo caminhos para léxicos
        """
        self.config = config or {}
        self.lexicons = {}
        self.default_lexicon = self._create_default_lexicon()
        
        # Carregar léxicos configurados
        lexicon_path = self.config.get("lexicon_path")
        if lexicon_path:
            self.load_lexicon(lexicon_path)
            
    def _create_default_lexicon(self) -> Dict[str, List[str]]:
        """
        Cria um léxico financeiro básico como fallback.
        
        Returns:
            Léxico básico com termos positivos, negativos e neutros
        """
        return {
            "positive": [
                "crescimento", "aumento", "lucro", "positivo", "alta", "valorização",
                "superou", "melhor", "expansão", "superar", "rentabilidade", "benefício",
                "apreciação", "excelente", "eficiência", "sucesso", "oportunidade", "forte"
            ],
            "negative": [
                "queda", "redução", "prejuízo", "negativo", "baixa", "desvalorização",
                "perda", "pior", "contração", "diminuição", "risco", "preocupação",
                "deterioração", "fraco", "abaixo", "ameaça", "dívida", "déficit"
            ],
            "neutral": [
                "estável", "manteve", "constante", "igual", "mesmo", "padrão",
                "esperado", "em linha", "conforme", "similar", "comparável", "manter",
                "previsto", "consistente", "sem alteração", "normal", "regular"
            ]
        }
        
    def load_lexicon(self, path: str) -> bool:
        """
        Carrega um léxico a partir de um arquivo.
        
        Args:
            path: Caminho para o arquivo de léxico (JSON ou CSV)
            
        Returns:
            True se o léxico foi carregado com sucesso, False caso contrário
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Arquivo de léxico não encontrado: {path}")
                return False
                
            if path.lower().endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    lexicon = json.load(f)
                    
            elif path.lower().endswith('.csv'):
                lexicon = {"positive": [], "negative": [], "neutral": []}
                with open(path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if "term" in row and "sentiment" in row:
                            sentiment = row["sentiment"].lower()
                            if sentiment in lexicon:
                                lexicon[sentiment].append(row["term"])
                                
            else:
                logger.error(f"Formato de arquivo de léxico não suportado: {path}")
                return False
                
            # Validar léxico carregado
            if not all(key in lexicon for key in ["positive", "negative", "neutral"]):
                logger.warning(f"Léxico no formato incorreto. Deve conter chaves: positive, negative, neutral")
                return False
                
            # Armazenar léxico
            filename = os.path.basename(path)
            self.lexicons[filename] = lexicon
            logger.info(f"Léxico carregado com sucesso: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao carregar léxico de {path}: {str(e)}")
            return False
            
    def get_lexicon(self, name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Obtém um léxico pelo nome.
        
        Args:
            name: Nome do léxico (ou None para o léxico padrão)
            
        Returns:
            Léxico de sentimento
        """
        if not name:
            # Se não houver léxicos carregados, retornar o léxico padrão
            if not self.lexicons:
                return self.default_lexicon
            # Caso contrário, retornar o primeiro léxico carregado
            return next(iter(self.lexicons.values()))
            
        # Retornar léxico específico, se existir
        if name in self.lexicons:
            return self.lexicons[name]
            
        # Fallback para o léxico padrão
        logger.warning(f"Léxico '{name}' não encontrado. Usando léxico padrão.")
        return self.default_lexicon
        
    def extend_lexicon(self, terms: Dict[str, List[str]], name: Optional[str] = None) -> bool:
        """
        Estende um léxico existente com novos termos.
        
        Args:
            terms: Dicionário com novos termos por categoria
            name: Nome do léxico a estender (None para o léxico padrão)
            
        Returns:
            True se o léxico foi estendido com sucesso, False caso contrário
        """
        lexicon = self.get_lexicon(name)
        
        try:
            for category, category_terms in terms.items():
                if category in lexicon:
                    # Adicionar novos termos (evitando duplicatas)
                    lexicon[category].extend([term for term in category_terms if term not in lexicon[category]])
                    logger.info(f"Léxico '{name or 'padrão'}' estendido com {len(category_terms)} termos na categoria '{category}'")
                else:
                    logger.warning(f"Categoria '{category}' não encontrada no léxico")
            return True
        except Exception as e:
            logger.error(f"Erro ao estender léxico: {str(e)}")
            return False
            
    def save_lexicon(self, path: str, name: Optional[str] = None) -> bool:
        """
        Salva um léxico em um arquivo.
        
        Args:
            path: Caminho para o arquivo de saída
            name: Nome do léxico a salvar (None para o léxico padrão)
            
        Returns:
            True se o léxico foi salvo com sucesso, False caso contrário
        """
        lexicon = self.get_lexicon(name)
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            if path.lower().endswith('.json'):
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(lexicon, f, ensure_ascii=False, indent=2)
                    
            elif path.lower().endswith('.csv'):
                with open(path, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['term', 'sentiment'])
                    for sentiment, terms in lexicon.items():
                        for term in terms:
                            writer.writerow([term, sentiment])
                            
            else:
                logger.error(f"Formato de arquivo não suportado: {path}")
                return False
                
            logger.info(f"Léxico salvo com sucesso em: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar léxico em {path}: {str(e)}")
            return False
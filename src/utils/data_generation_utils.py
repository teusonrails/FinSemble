"""
Utilitários para geração de dados do sistema FinSemble.

Este módulo contém classes utilitárias para geração de dados,
incluindo gerenciamento de templates e manipulação de datas.
"""

import os
import json
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta

# Configuração de logging
logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Gerenciador de templates para geração de dados sintéticos.
    """
    
    def __init__(self, templates_path: Optional[str] = None, fallback_templates: Optional[Dict[str, List[str]]] = None):
        """
        Inicializa o gerenciador de templates.
        
        Args:
            templates_path: Caminho para o arquivo de templates (opcional)
            fallback_templates: Templates padrão a usar se arquivo não existir (opcional)
        """
        self.templates_path = templates_path
        self.fallback_templates = fallback_templates or {}
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, List[str]]:
        """
        Carrega os templates de texto para cada categoria.
        
        Returns:
            Dicionário com templates por categoria
        """
        # Templates padrão se não encontrar os arquivos
        default_templates = self.fallback_templates
        
        if not default_templates:
            # Templates mínimos se não houver fallback configurado
            default_templates = {
                "relatorio_trimestral": [
                    "A {company} ({stock_code}) divulga seus resultados referentes ao {quarter} trimestre de {year}."
                ],
                "comunicado_mercado": [
                    "COMUNICADO AO MERCADO: A {company} informa aos seus acionistas e ao mercado em geral que {announcement_content}."
                ]
            }
        
        try:
            # Tentar carregar de arquivo, se existir
            if self.templates_path and os.path.exists(self.templates_path):
                with open(self.templates_path, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
                logger.info(f"Templates carregados de {self.templates_path}")
                return templates
        except Exception as e:
            logger.warning(f"Erro ao carregar templates de arquivo: {str(e)}")
        
        logger.info("Usando templates padrão")
        return default_templates
    
    def get_templates(self, category: str) -> List[str]:
        """
        Obtém os templates de uma categoria específica.
        
        Args:
            category: Categoria dos templates desejados
            
        Returns:
            Lista de templates da categoria, ou lista vazia se categoria não existir
        """
        return self.templates.get(category, [])
    
    def get_random_template(self, category: str) -> Optional[str]:
        """
        Obtém um template aleatório de uma categoria.
        
        Args:
            category: Categoria do template desejado
            
        Returns:
            Template aleatório, ou None se categoria não existir ou estiver vazia
        """
        templates = self.get_templates(category)
        return random.choice(templates) if templates else None
    
    def get_all_categories(self) -> List[str]:
        """
        Obtém todas as categorias disponíveis.
        
        Returns:
            Lista de categorias
        """
        return list(self.templates.keys())


class DateUtils:
    """
    Utilitários para geração e manipulação de datas.
    """
    
    @staticmethod
    def generate_date_range(base_date=None, days_back: int = 365*3, 
                          is_business_day: bool = True) -> datetime:
        """
        Gera uma data aleatória dentro de um intervalo.
        
        Args:
            base_date: Data base (padrão: data atual)
            days_back: Número máximo de dias para voltar
            is_business_day: Se deve garantir que seja um dia útil
            
        Returns:
            Data gerada
        """
        # Data base
        if base_date is None:
            base_date = datetime.now()
            
        # Gerar data aleatória dentro do intervalo
        random_days = random.randint(0, days_back)
        date = base_date - timedelta(days=random_days)
        
        # Ajustar para dia útil, se necessário
        if is_business_day:
            # Ajustar para um dia útil (não fim de semana)
            while date.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
                date -= timedelta(days=1)
        
        return date
    
    @staticmethod
    def generate_business_dates(base_date=None, 
                              meeting_days_offset: Tuple[int, int] = (-30, -5),
                              record_days_offset: Tuple[int, int] = (5, 20),
                              payment_days_offset: Tuple[int, int] = (25, 50)) -> Dict[str, str]:
        """
        Gera um conjunto de datas relacionadas para uso em templates financeiros.
        
        Args:
            base_date: Data base para a geração (padrão: data aleatória nos últimos 3 anos)
            meeting_days_offset: Intervalo de dias para data de reunião (relativo à data base)
            record_days_offset: Intervalo de dias para data de registro (relativo à data base)
            payment_days_offset: Intervalo de dias para data de pagamento (relativo à data de registro)
            
        Returns:
            Dicionário com diferentes datas formatadas
        """
        # Gerar data base aleatória, se não fornecida
        if base_date is None:
            base_date = DateUtils.generate_date_range()
        
        # Criar variações de datas
        meeting_offset = random.randint(*meeting_days_offset)
        meeting_date = base_date + timedelta(days=meeting_offset)
        
        # Ajustar para dia útil, se necessário
        while meeting_date.weekday() >= 5:
            meeting_date -= timedelta(days=1)
            
        record_offset = random.randint(*record_days_offset)
        record_date = base_date + timedelta(days=record_offset)
        
        # Ajustar para dia útil, se necessário
        while record_date.weekday() >= 5:
            record_date -= timedelta(days=1)
            
        payment_offset = random.randint(*payment_days_offset)
        payment_date = record_date + timedelta(days=payment_offset)
        
        # Ajustar para dia útil, se necessário
        while payment_date.weekday() >= 5:
            payment_date -= timedelta(days=1)
        
        # Formatar datas
        format_date = lambda d: d.strftime("%d/%m/%Y")
        
        # Informações de trimestre e ano
        quarter = (base_date.month - 1) // 3 + 1
        year = base_date.year
        
        # Período de referência
        period_options = [
            f"{quarter}º trimestre de {year}",
            f"ano fiscal de {year}",
            f"primeiro semestre de {year}" if quarter <= 2 else f"segundo semestre de {year}"
        ]
        period = random.choice(period_options)
        
        return {
            "date": format_date(base_date),
            "meeting_date": format_date(meeting_date),
            "record_date": format_date(record_date),
            "payment_date": format_date(payment_date),
            "quarter": str(quarter),
            "year": str(year),
            "period": period
        }
"""
Módulo de validação de inputs para o FinSemble.

Este módulo contém funções e classes para validação e sanitização
de entradas, garantindo que o sistema receba dados consistentes e
tratando adequadamente casos de borda e inputs atípicos.
"""

import re
import json
import logging
import datetime
from typing import Any, Dict, List, Union, Optional, Tuple, Set

# Configuração de logging
logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Exceção lançada quando um input falha na validação."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        """
        Inicializa uma exceção de validação.
        
        Args:
            message: Mensagem de erro
            field: Campo que falhou na validação (opcional)
            value: Valor que causou o erro (opcional)
        """
        self.field = field
        self.value = value
        self.message = message
        
        detail = ""
        if field:
            detail += f" Campo: '{field}'."
        if value is not None:
            # Limitar tamanho do valor para evitar mensagens de erro muito grandes
            str_value = str(value)
            if len(str_value) > 100:
                str_value = str_value[:97] + "..."
            detail += f" Valor: '{str_value}'."
            
        super().__init__(message + detail)


class InputValidator:
    """
    Classe para validação e sanitização de inputs.
    
    Esta classe fornece métodos para validar e sanitizar diferentes
    tipos de entradas, como textos, metadados e outras estruturas
    de dados utilizadas pelo sistema FinSemble.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Inicializa o validador com a configuração especificada.
        
        Args:
            config: Configurações para validação, incluindo regras
                   específicas para diferentes tipos de dados.
        """
        self.config = config or {}
        
        # Configurações específicas para validação de texto
        self.text_config = self.config.get("text", {})
        self.min_text_length = self.text_config.get("min_length", 1)
        self.max_text_length = self.text_config.get("max_length", 1000000)
        self.allow_empty = self.text_config.get("allow_empty", False)
        
        # Configurações para metadados
        self.metadata_config = self.config.get("metadata", {})
        self.required_metadata_fields = self.metadata_config.get("required_fields", [])
        self.allowed_metadata_fields = self.metadata_config.get("allowed_fields")  # None = sem restrições
    
    def validate_text(self, text: Any) -> str:
        """
        Valida e sanitiza um texto de entrada.
        
        Args:
            text: Texto a ser validado
            
        Returns:
            Texto sanitizado
            
        Raises:
            ValidationError: Se o texto for inválido
        """
        # Verificação de tipo
        if text is None:
            if self.allow_empty:
                return ""
            raise ValidationError("Texto não pode ser None", value=text)
            
        if not isinstance(text, str):
            # Tentar converter para string
            try:
                text = str(text)
                logger.warning(f"Texto não era do tipo str. Convertido para: {text[:100]}...")
            except:
                raise ValidationError("Texto deve ser uma string", value=type(text))
        
        # Verificação de comprimento
        if len(text) < self.min_text_length and not (self.allow_empty and len(text) == 0):
            raise ValidationError(
                f"Texto muito curto. Mínimo: {self.min_text_length} caracteres", 
                value=text
            )
            
        if len(text) > self.max_text_length:
            logger.warning(f"Texto truncado de {len(text)} para {self.max_text_length} caracteres")
            text = text[:self.max_text_length]
        
        # Sanitização
        # Remover caracteres de controle exceto quebras de linha e tabs
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    
    def validate_metadata(self, metadata: Any) -> Dict[str, Any]:
        """
        Valida e sanitiza metadados.
        
        Args:
            metadata: Metadados a serem validados
            
        Returns:
            Metadados sanitizados
            
        Raises:
            ValidationError: Se os metadados forem inválidos
        """
        # Verificação de tipo
        if metadata is None:
            return {}
            
        if not isinstance(metadata, dict):
            try:
                # Tentar converter de string JSON
                if isinstance(metadata, str):
                    metadata = json.loads(metadata)
                else:
                    raise ValidationError("Metadados devem ser um dicionário", value=type(metadata))
            except json.JSONDecodeError:
                raise ValidationError("Metadados em formato JSON inválido", value=metadata)
        
        # Verificar campos obrigatórios
        for field in self.required_metadata_fields:
            if field not in metadata:
                raise ValidationError(f"Campo obrigatório ausente nos metadados", field=field)
        
        # Restringir a campos permitidos, se configurado
        if self.allowed_metadata_fields is not None:
            invalid_fields = [field for field in metadata if field not in self.allowed_metadata_fields]
            if invalid_fields:
                logger.warning(f"Campos não permitidos removidos dos metadados: {invalid_fields}")
                metadata = {k: v for k, v in metadata.items() if k in self.allowed_metadata_fields}
        
        # Sanitizar valores
        sanitized = {}
        for key, value in metadata.items():
            # Converter datas para strings ISO
            if isinstance(value, (datetime.date, datetime.datetime)):
                sanitized[key] = value.isoformat()
            # Sanitizar strings dentro dos metadados
            elif isinstance(value, str):
                sanitized[key] = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', value)
            # Limitar profundidade de dicionários aninhados
            elif isinstance(value, dict) and len(str(value)) > 1000:
                logger.warning(f"Metadados aninhados muito extensos. Truncando: {key}")
                sanitized[key] = "aninhamento_truncado"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def validate_batch_inputs(self, 
                             texts: List[Any], 
                             metadatas: Optional[List[Any]] = None,
                             max_batch_size: Optional[int] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Valida e sanitiza lotes de textos e metadados.
        
        Args:
            texts: Lista de textos
            metadatas: Lista de metadados (opcional)
            max_batch_size: Tamanho máximo do lote (opcional)
            
        Returns:
            Tupla (textos_sanitizados, metadados_sanitizados)
            
        Raises:
            ValidationError: Se houver erros de validação
        """
        if not isinstance(texts, (list, tuple)):
            raise ValidationError("Lista de textos deve ser uma lista ou tupla", value=type(texts))
        
        if max_batch_size and len(texts) > max_batch_size:
            logger.warning(f"Lote muito grande ({len(texts)}). Truncando para {max_batch_size} itens.")
            texts = texts[:max_batch_size]
            if metadatas and len(metadatas) > max_batch_size:
                metadatas = metadatas[:max_batch_size]
        
        # Sanitizar cada texto
        sanitized_texts = []
        for i, text in enumerate(texts):
            try:
                sanitized_texts.append(self.validate_text(text))
            except ValidationError as e:
                logger.warning(f"Texto inválido no índice {i}: {str(e)}")
                sanitized_texts.append("")  # Usar string vazia como fallback
        
        # Sanitizar metadados, se fornecidos
        sanitized_metadatas = []
        if metadatas:
            if len(metadatas) < len(texts):
                logger.warning(f"Número de metadados ({len(metadatas)}) menor que textos ({len(texts)}). Preenchendo com dicionários vazios.")
                metadatas = metadatas + [{}] * (len(texts) - len(metadatas))
            elif len(metadatas) > len(texts):
                logger.warning(f"Número de metadados ({len(metadatas)}) maior que textos ({len(texts)}). Truncando.")
                metadatas = metadatas[:len(texts)]
                
            for i, metadata in enumerate(metadatas):
                try:
                    sanitized_metadatas.append(self.validate_metadata(metadata))
                except ValidationError as e:
                    logger.warning(f"Metadados inválidos no índice {i}: {str(e)}")
                    sanitized_metadatas.append({})  # Usar dicionário vazio como fallback
        else:
            # Se não houver metadados, usar dicionários vazios
            sanitized_metadatas = [{} for _ in range(len(sanitized_texts))]
        
        return sanitized_texts, sanitized_metadatas
    
    def validate_config(self, config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida a configuração do sistema contra um esquema predefinido.
        
        Args:
            config: Configuração a ser validada
            schema: Esquema de validação
            
        Returns:
            Configuração validada
            
        Raises:
            ValidationError: Se a configuração for inválida
        """
        if not isinstance(config, dict):
            raise ValidationError("Configuração deve ser um dicionário", value=type(config))
            
        # Verificar campos obrigatórios
        for field, field_schema in schema.items():
            if field_schema.get("required", False) and field not in config:
                raise ValidationError(f"Campo obrigatório ausente na configuração", field=field)
        
        # Validar valores com base no tipo esperado
        for field, value in config.items():
            if field in schema:
                field_schema = schema[field]
                expected_type = field_schema.get("type")
                
                # Verificar tipo
                if expected_type and not isinstance(value, expected_type):
                    raise ValidationError(
                        f"Tipo inválido para campo na configuração. Esperado: {expected_type.__name__}",
                        field=field,
                        value=value
                    )
                
                # Verificar valores permitidos
                allowed_values = field_schema.get("allowed_values")
                if allowed_values and value not in allowed_values:
                    raise ValidationError(
                        f"Valor não permitido para campo. Permitidos: {allowed_values}",
                        field=field,
                        value=value
                    )
                
                # Validar recursivamente configurações aninhadas
                if isinstance(value, dict) and "properties" in field_schema:
                    sub_schema = field_schema["properties"]
                    config[field] = self.validate_config(value, sub_schema)
        
        return config


# Função de fábrica para criar validadores
def create_validator(config: Dict[str, Any] = None) -> InputValidator:
    """
    Cria um validador com a configuração especificada.
    
    Args:
        config: Configurações para o validador
        
    Returns:
        Instância de InputValidator
    """
    if config is None:
        # Carregar configuração padrão
        from src.utils.config import load_config, get_config_section
        try:
            full_config = load_config()
            validation_config = get_config_section(full_config, "validation")
        except:
            logger.warning("Erro ao carregar configuração. Usando configurações padrão.")
            validation_config = {}
    else:
        validation_config = config
        
    return InputValidator(validation_config)


# Funções auxiliares para validação rápida
def sanitize_text(text: Any) -> str:
    """
    Sanitiza um texto rapidamente.
    
    Args:
        text: Texto a ser sanitizado
        
    Returns:
        Texto sanitizado
    """
    validator = create_validator()
    try:
        return validator.validate_text(text)
    except ValidationError as e:
        logger.warning(f"Erro ao sanitizar texto: {str(e)}")
        return "" if text is None else str(text)


def sanitize_metadata(metadata: Any) -> Dict[str, Any]:
    """
    Sanitiza metadados rapidamente.
    
    Args:
        metadata: Metadados a serem sanitizados
        
    Returns:
        Metadados sanitizados
    """
    validator = create_validator()
    try:
        return validator.validate_metadata(metadata)
    except ValidationError as e:
        logger.warning(f"Erro ao sanitizar metadados: {str(e)}")
        return {}
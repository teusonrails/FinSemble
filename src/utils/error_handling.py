"""
Módulo para tratamento unificado de erros no sistema FinSemble.

Este módulo fornece classes e funções para garantir um tratamento
consistente de erros em todos os componentes do sistema.
"""

import logging
import traceback
from typing import Dict, Any, Optional, Type, List
from enum import Enum

# Configuração de logging
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Níveis de severidade para erros no sistema."""
    CRITICAL = "critical"    # Erros que impedem completamente a operação
    ERROR = "error"          # Erros graves que afetam a funcionalidade principal
    WARNING = "warning"      # Problemas que podem afetar a qualidade dos resultados
    INFO = "info"            # Informações que não são erros, mas merecem atenção


class ErrorCode(Enum):
    """Códigos de erro padronizados para o sistema FinSemble."""
    # Erros gerais
    UNKNOWN_ERROR = "unknown_error"
    INTERNAL_ERROR = "internal_error"
    NOT_IMPLEMENTED = "not_implemented"
    
    # Erros de inicialização
    INITIALIZATION_ERROR = "initialization_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    CONFIG_ERROR = "configuration_error"
    
    # Erros de dados de entrada
    INVALID_INPUT = "invalid_input"
    EMPTY_INPUT = "empty_input"
    MALFORMED_INPUT = "malformed_input"
    
    # Erros de modelo
    MODEL_NOT_TRAINED = "model_not_trained"
    MODEL_LOAD_ERROR = "model_load_error"
    MODEL_SAVE_ERROR = "model_save_error"
    
    # Erros de processamento
    PREPROCESSING_ERROR = "preprocessing_error"
    FEATURE_EXTRACTION_ERROR = "feature_extraction_error"
    PREDICTION_ERROR = "prediction_error"
    CLASSIFICATION_ERROR = "classification_error"
    
    # Erros de inferência bayesiana
    INFERENCE_ERROR = "inference_error"
    CONVERGENCE_ERROR = "convergence_error"
    INCOMPATIBLE_EVIDENCE = "incompatible_evidence"
    
    # Erros de validação
    VALIDATION_ERROR = "validation_error"
    DIMENSIONALITY_ERROR = "dimensionality_error"


class FinSembleException(Exception):
    """
    Classe base para todas as exceções específicas do FinSemble.
    Permite um tratamento consistente de erros em todo o sistema.
    """
    
    def __init__(self, 
                message: str, 
                code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                severity: ErrorSeverity = ErrorSeverity.ERROR,
                component: str = None,
                details: Dict[str, Any] = None):
        """
        Inicializa a exceção com informações detalhadas.
        
        Args:
            message: Mensagem de erro descritiva
            code: Código de erro da enumeração ErrorCode
            severity: Nível de severidade do erro
            component: Componente onde o erro ocorreu
            details: Detalhes adicionais sobre o erro
        """
        self.message = message
        self.code = code
        self.severity = severity
        self.component = component
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a exceção para um dicionário, útil para APIs e logging.
        
        Returns:
            Dicionário representando o erro
        """
        return {
            "error": True,
            "error_code": self.code.value,
            "error_message": self.message,
            "severity": self.severity.value,
            "component": self.component,
            "details": self.details
        }


class ErrorHandler:
    """
    Gerenciador centralizado para tratamento de erros.
    
    Esta classe fornece métodos para lidar com erros de maneira consistente
    em todos os componentes do sistema FinSemble.
    """
    
    @staticmethod
    def log_exception(exception: Exception, component: str, 
                     log_traceback: bool = True) -> None:
        """
        Registra uma exceção no log de maneira padronizada.
        
        Args:
            exception: A exceção a ser registrada
            component: O componente onde a exceção ocorreu
            log_traceback: Se deve incluir o traceback completo
        """
        error_message = f"Erro em {component}: {str(exception)}"
        
        if isinstance(exception, FinSembleException):
            log_method = logger.error
            if exception.severity == ErrorSeverity.CRITICAL:
                log_method = logger.critical
            elif exception.severity == ErrorSeverity.WARNING:
                log_method = logger.warning
            elif exception.severity == ErrorSeverity.INFO:
                log_method = logger.info
                
            log_method(f"{error_message} [Código: {exception.code.value}]")
            
            if exception.details:
                logger.debug(f"Detalhes do erro: {exception.details}")
        else:
            logger.error(error_message)
            
        if log_traceback:
            logger.debug("Traceback: %s", traceback.format_exc())
    
    @staticmethod
    def handle_exception(exception: Exception, component: str, 
                        fallback_enabled: bool = True) -> Dict[str, Any]:
        """
        Manipula uma exceção e retorna um dicionário de erro padronizado.
        
        Args:
            exception: A exceção a ser tratada
            component: O componente onde a exceção ocorreu
            fallback_enabled: Se deve tentar recuperar-se do erro
            
        Returns:
            Dicionário com informações de erro padronizadas
        """
        # Registrar a exceção
        ErrorHandler.log_exception(exception, component)
        
        # Converter para formato padronizado
        if isinstance(exception, FinSembleException):
            error_dict = exception.to_dict()
        else:
            error_dict = {
                "error": True,
                "error_code": ErrorCode.UNKNOWN_ERROR.value,
                "error_message": str(exception),
                "severity": ErrorSeverity.ERROR.value,
                "component": component
            }
            
        # Adicionar traceback em desenvolvimento
        if logger.level <= logging.DEBUG:
            error_dict["traceback"] = traceback.format_exc()
            
        return error_dict
    
    @staticmethod
    def create_exception(message: str, 
                       code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                       severity: ErrorSeverity = ErrorSeverity.ERROR,
                       component: str = None,
                       details: Dict[str, Any] = None) -> FinSembleException:
        """
        Cria uma exceção FinSemble padronizada.
        
        Args:
            message: Mensagem de erro
            code: Código de erro
            severity: Severidade do erro
            component: Componente onde o erro ocorreu
            details: Detalhes adicionais
            
        Returns:
            Instância de FinSembleException
        """
        return FinSembleException(
            message=message,
            code=code,
            severity=severity,
            component=component,
            details=details
        )
        
    @staticmethod
    def raiseif(condition: bool, message: str, 
              code: ErrorCode = ErrorCode.VALIDATION_ERROR,
              severity: ErrorSeverity = ErrorSeverity.ERROR,
              component: str = None,
              details: Dict[str, Any] = None) -> None:
        """
        Levanta uma exceção se a condição for verdadeira.
        
        Args:
            condition: Condição a verificar
            message: Mensagem de erro
            code: Código de erro
            severity: Severidade do erro
            component: Componente onde o erro ocorreu
            details: Detalhes adicionais
            
        Raises:
            FinSembleException: Se a condição for verdadeira
        """
        if condition:
            raise ErrorHandler.create_exception(
                message=message,
                code=code,
                severity=severity,
                component=component,
                details=details
            )
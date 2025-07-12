"""
Logging utilities for ICA Framework
"""

import sys
import logging
from typing import Optional
from pathlib import Path

# Try to import loguru, fall back to standard logging if not available
try:
    from loguru import logger as loguru_logger
    HAS_LOGURU = True
except ImportError:
    HAS_LOGURU = False
    loguru_logger = None


class Logger:
    """Custom logger for ICA Framework with fallback support"""
    
    def __init__(self, name: str = "ICA", level: str = "INFO", log_file: Optional[str] = None):
        self.name = name
        self.level = level
        self.log_file = log_file
        self.use_loguru = HAS_LOGURU
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with loguru or standard logging"""
        if self.use_loguru:
            self._setup_loguru()
        else:
            self._setup_standard_logging()
    
    def _setup_loguru(self):
        """Setup loguru logger with custom configuration"""
        # Remove default handler
        loguru_logger.remove()
        
        # Add console handler
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.level,
            colorize=True
        )
        
        # Add file handler if specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            loguru_logger.add(
                self.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=self.level,
                rotation="100 MB",
                retention="30 days",
                compression="zip"
            )
    
    def _setup_standard_logging(self):
        """Setup standard Python logging as fallback"""
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(getattr(logging, self.level.upper()))
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stderr)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if self.log_file:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(self.log_file)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.info(formatted_message, **kwargs)
        else:
            self.logger.info(formatted_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.debug(formatted_message, **kwargs)
        else:
            self.logger.debug(formatted_message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.warning(formatted_message, **kwargs)
        else:
            self.logger.warning(formatted_message)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.error(formatted_message, **kwargs)
        else:
            self.logger.error(formatted_message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.critical(formatted_message, **kwargs)
        else:
            self.logger.critical(formatted_message)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        formatted_message = f"[{self.name}] {message}"
        if self.use_loguru:
            loguru_logger.exception(formatted_message, **kwargs)
        else:
            self.logger.exception(formatted_message)


# Global logger instance
ica_logger = Logger("ICA-Framework")

"""
Logger Module

Sets up structured logging with file rotation and colored console output.
"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

try:
    from rich.logging import RichHandler
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str = "subtitle_retrieval",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Setup logger with file rotation and colored console output.
    
    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, no file logging.
        console_output: Enable console output
        file_output: Enable file output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup log files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    if console_output:
        if RICH_AVAILABLE:
            # Use Rich handler for better formatting
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                markup=True,
            )
            console_handler.setLevel(numeric_level)
            logger.addHandler(console_handler)
        else:
            # Fallback to colored formatter
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(numeric_level)
            formatter = ColoredFormatter(
                fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output and log_file:
        log_path = Path(log_file)
        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get logger instance. If name is provided, returns logger with that name.
    Otherwise returns default logger.
    
    Args:
        name: Logger name. If None, uses default name.
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logging.getLogger("subtitle_retrieval")


# Default logger instance (will be initialized when config is available)
_default_logger: Optional[logging.Logger] = None


def init_logger_from_config(config) -> logging.Logger:
    """
    Initialize logger from configuration object.
    
    Args:
        config: Configuration object with LOG_LEVEL and LOG_FILE attributes
    
    Returns:
        Configured logger instance
    """
    global _default_logger
    _default_logger = setup_logger(
        name="subtitle_retrieval",
        log_level=config.LOG_LEVEL,
        log_file=config.LOG_FILE,
    )
    return _default_logger


def get_default_logger() -> logging.Logger:
    """
    Get default logger instance. Initializes if not already initialized.
    
    Returns:
        Logger instance
    """
    global _default_logger
    if _default_logger is None:
        # Try to initialize from config if available
        try:
            from .config import get_config
            config = get_config()
            _default_logger = init_logger_from_config(config)
        except Exception:
            # Fallback to basic logger
            _default_logger = setup_logger(
                name="subtitle_retrieval",
                log_level="INFO",
                log_file=None,
            )
    return _default_logger

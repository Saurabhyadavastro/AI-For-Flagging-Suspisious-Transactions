"""Logging configuration and utilities."""

import logging
import sys
from pathlib import Path
from typing import Optional

# Try to import loguru, fall back to standard logging if not available
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    # Create a simple logger that mimics loguru interface
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Add console handler if not already present
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# Remove default loguru logger if available
if LOGURU_AVAILABLE:
    logger.remove()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    include_console: bool = True
) -> None:
    """Set up logging configuration with loguru or standard logging."""
    
    if LOGURU_AVAILABLE:
        # Console logging with loguru
        if include_console:
            logger.add(
                sys.stderr,
                level=log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
        
        # File logging with loguru
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="10 MB",
                retention="1 month",
                compression="zip"
            )
    else:
        # Standard logging setup
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
            handlers=[
                logging.StreamHandler(sys.stderr) if include_console else None,
                logging.FileHandler(log_file) if log_file else None
            ]
        )


def get_logger(name: str):
    """Get a logger instance with the given name."""
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    else:
        return logging.getLogger(name)


# Set up default logging
setup_logging()


class InterceptHandler(logging.Handler):
    """Intercept standard logging and redirect to loguru."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record."""
        if not LOGURU_AVAILABLE:
            return
            
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_standard_logging_intercept() -> None:
    """Set up interception of standard library logging."""
    if not LOGURU_AVAILABLE:
        return
        
    # Intercept everything at the root logger
    logging.root.handlers = [InterceptHandler()]
    logging.root.setLevel(logging.DEBUG)
    
    # Remove every other logger's handlers and propagate to root logger
    for name in logging.root.manager.loggerDict.keys():
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True


# Initialize standard logging interception if loguru is available
if LOGURU_AVAILABLE:
    setup_standard_logging_intercept()

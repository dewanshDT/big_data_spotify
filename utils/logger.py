"""
Simple logging utility for all scripts
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with both file and console output
    
    Args:
        name: Logger name (usually __name__)
        log_file: Path to log file (optional, defaults to logs/{name}.log)
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger instance
    """
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Setup log file name
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f"{name}_{timestamp}.log"
    else:
        log_file = logs_dir / log_file
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # File handler (detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simpler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def log_dataframe_info(logger, df, name="DataFrame"):
    """Log DataFrame information"""
    logger.info(f"{name} shape: {df.shape}")
    logger.info(f"{name} memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    logger.info(f"{name} columns: {list(df.columns)}")

def log_progress(logger, current, total, message="Processing"):
    """Log progress"""
    percentage = (current / total) * 100
    logger.info(f"{message}: {current}/{total} ({percentage:.1f}%)")


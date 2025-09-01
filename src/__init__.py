"""
AI For Flagging Suspicious Transactions

An AI-powered system for detecting suspicious transactions in Indian Rupee payments.
"""

__version__ = "0.1.0"
__author__ = "AI Project Team"
__email__ = "team@transaction-anomaly-detector.com"

# Core imports for easy access
from src.utils.config import Config
from src.utils.logger import get_logger

__all__ = ["Config", "get_logger"]

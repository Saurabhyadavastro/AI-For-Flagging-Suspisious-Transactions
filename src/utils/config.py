"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    """Application configuration using Pydantic BaseSettings."""
    
    # Application settings
    app_name: str = Field(default="AI For Flagging Suspicious Transactions")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)
    
    # Database settings
    supabase_url: Optional[str] = Field(default=None)
    supabase_key: Optional[str] = Field(default=None)
    database_url: Optional[str] = Field(default=None)
    
    # ML Model settings
    model_path: str = Field(default="data/models")
    max_features: int = Field(default=1000)
    anomaly_threshold: float = Field(default=0.5)
    
    # Ollama settings
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1")
    
    # Streamlit settings
    streamlit_host: str = Field(default="localhost")
    streamlit_port: int = Field(default=8501)
    
    # Security settings
    secret_key: str = Field(default="dev-secret-key-change-in-production")
    encryption_key: Optional[str] = Field(default=None)
    
    # Logging settings
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/app.log")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "allow"  # Allow extra fields from environment


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return {}
    
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def get_config() -> Config:
    """Get the application configuration instance."""
    return Config()


# Global configuration instance
config = get_config()

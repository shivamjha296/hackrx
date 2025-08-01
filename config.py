"""
Configuration settings for HackRX 5.0 application
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("DEBUG", "True").lower() == "true"
    
    # Authentication
    BEARER_TOKEN = os.getenv("BEARER_TOKEN", "your-bearer-token-here")
    
    # Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
    GEMINI_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-lite")
    
    # Document Processing
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = ['.pdf', '.doc', '.docx', '.eml', '.msg']
    DOWNLOAD_TIMEOUT = 30  # seconds
    
    # Text Processing
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Vector Store
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))
    
    # Performance
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
    REQUEST_TIMEOUT = 300  # 5 minutes
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def get_device(cls):
        """Get the computing device (cuda/cpu)"""
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        if not cls.GEMINI_API_KEY or cls.GEMINI_API_KEY == "your-gemini-api-key-here":
            raise ValueError("GEMINI_API_KEY must be set in environment variables or .env file")
        
        if not cls.BEARER_TOKEN or cls.BEARER_TOKEN == "your-bearer-token-here":
            raise ValueError("BEARER_TOKEN must be set in environment variables or .env file")
        
        return True

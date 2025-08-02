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
    
    # Groq API
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")  # Fast model for low latency
    
    # Document Processing
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = ['.pdf', '.doc', '.docx', '.eml', '.msg']
    DOWNLOAD_TIMEOUT = 30  # seconds
    
    # Text Processing (Optimized for speed)
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))  # Smaller chunks for faster processing
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # Reduced overlap for speed
    
    # Embeddings
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
    
    # Vector Store (Optimized for speed)
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "3"))  # Fewer documents for faster retrieval
    
    # Performance (Optimized for low latency)
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Smaller batch size for faster processing
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))  # Reduced timeout
    
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
        if not cls.GROQ_API_KEY or cls.GROQ_API_KEY == "your-groq-api-key-here":
            raise ValueError("GROQ_API_KEY must be set in environment variables or .env file")
        
        if not cls.BEARER_TOKEN or cls.BEARER_TOKEN == "your-bearer-token-here":
            raise ValueError("BEARER_TOKEN must be set in environment variables or .env file")
        
        return True

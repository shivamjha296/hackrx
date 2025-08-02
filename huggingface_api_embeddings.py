"""
Hugging Face API Embeddings Implementation
Replaces local sentence-transformers with API calls to Hugging Face Inference API
"""

import requests
import time
import logging
from typing import List, Optional
from langchain.embeddings.base import Embeddings
import os

logger = logging.getLogger(__name__)

class HuggingFaceAPIEmbeddings(Embeddings):
    """
    Hugging Face API Embeddings using remote inference instead of local models
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",  # Better API compatibility
        api_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.model_name = model_name
        self.api_token = api_token or os.getenv("HUGGINGFACE_API_TOKEN")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        if not self.api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN must be set in environment variables")
        
        # Hugging Face Inference API endpoint - correct format
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"Initialized HuggingFace API embeddings with model: {self.model_name}")
    
    def _make_request(self, texts: List[str]) -> List[List[float]]:
        """Make API request to Hugging Face with retry logic"""
        # Use simple inputs format for feature extraction models
        payload = {"inputs": texts}
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Handle different response formats
                    if isinstance(result, list):
                        # Check if it's a list of embeddings
                        if len(result) > 0 and isinstance(result[0], list):
                            return result
                        # If it's a single embedding, wrap it
                        elif len(result) > 0 and isinstance(result[0], (int, float)):
                            return [result]
                    
                    logger.error(f"Unexpected response format: {type(result)}")
                    raise ValueError("Unexpected response format from Hugging Face API")
                
                elif response.status_code == 503:
                    # Model is loading, wait and retry
                    logger.warning(f"Model loading, retrying in {self.retry_delay * (attempt + 1)} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
                    
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise Exception(f"All API requests failed: {str(e)}")
        
        raise Exception("Max retries exceeded")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using Hugging Face API"""
        if not texts:
            return []
        
        try:
            logger.info(f"Embedding {len(texts)} documents via Hugging Face API...")
            
            # Process in batches to avoid API limits
            batch_size = 10  # Adjust based on API limits
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                
                embeddings = self._make_request(batch)
                all_embeddings.extend(embeddings)
                
                # Small delay between batches to respect rate limits
                if i + batch_size < len(texts):
                    time.sleep(0.1)
            
            logger.info(f"Successfully embedded {len(all_embeddings)} documents")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using Hugging Face API"""
        try:
            logger.debug(f"Embedding query via Hugging Face API...")
            embeddings = self._make_request([text])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "provider": "huggingface_api",
            "model_name": self.model_name,
            "api_endpoint": self.api_url,
            "local_processing": False
        }

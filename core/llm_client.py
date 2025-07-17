# core/llm_client.py
"""
LLM Client Module
Handles communication with the OpenRouter API.
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional

from config import Config

logger = logging.getLogger(__name__)


class OpenRouterLLM:
    """OpenRouter LLM client for response generation"""
    
    def __init__(self, api_key: str):
        """Initialize OpenRouter client"""
        self.api_key = api_key
        self.base_url = Config.OPENROUTER_BASE_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info("OpenRouter LLM client initialized")
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         max_tokens: int = None, 
                         temperature: float = None) -> str:
        """Generate response using OpenRouter API"""
        
        payload = {
            "model": Config.LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens or Config.MAX_TOKENS,
            "temperature": temperature or Config.TEMPERATURE,
            "top_p": Config.TOP_P
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected API response format: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

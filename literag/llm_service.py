"""
LLM Service for LiteRAG

Handles LLM interactions for query expansion and response generation.
Uses Gemini Flash Lite by default.
"""

import os
import time
from typing import Optional, Dict, Any
import logging
import hashlib

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for LLM interactions.
    
    Supports:
    - Direct Gemini API
    - LiteLLM proxy (for flexibility)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini/gemini-flash-lite-latest",
        max_retries: int = 3,
        timeout: int = 30,
        use_litellm: bool = True,
        litellm_base_url: str = "http://localhost:4000"
    ):
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_litellm = use_litellm
        self.litellm_base_url = litellm_base_url
        
        # Track usage - separate prompt and completion tokens
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        
        # Cache for GenerativeModel instances to avoid cold start
        self._model_cache: Dict[str, Any] = {}
        
        self._init_client()

    def __repr__(self):
        """Safely mask the API key when printing the service configuration."""
        key_masked = f"{self.api_key[:4]}...{self.api_key[-4:]}" if self.api_key else "None"
        return f"<LLMService model={self.model} api_key={key_masked}>"
    
    def _init_client(self):
        """Initialize the LLM client."""
        if self.use_litellm:
            try:
                import litellm
                self._client_type = "litellm"
                logger.info(f"Using LiteLLM with model {self.model}")
            except ImportError:
                logger.warning("LiteLLM not available, falling back to direct API")
                self._init_direct_client()
        else:
            self._init_direct_client()
    
    def _init_direct_client(self):
        """Initialize direct Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._genai = genai
            self._client_type = "genai"
            logger.info("Using direct Gemini API")
        except ImportError:
            # Fallback to requests
            self._client_type = "requests"
            logger.info("Using direct API requests")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Generate a response from the LLM.
        """
        for attempt in range(self.max_retries):
            try:
                if self._client_type == "litellm":
                    response = self._generate_litellm(
                        prompt, system_prompt, temperature, max_tokens
                    )
                elif self._client_type == "genai":
                    response = self._generate_genai(
                        prompt, system_prompt, temperature, max_tokens
                    )
                else:
                    response = self._generate_requests(
                        prompt, system_prompt, temperature, max_tokens
                    )
                
                self.call_count += 1
                logger.debug(
                    f"LLM call #{self.call_count}: prompt_tokens={self.prompt_tokens}, "
                    f"completion_tokens={self.completion_tokens}"
                )
                return response
                
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("Failed to generate response after retries")
    
    def _generate_litellm(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using LiteLLM."""
        import litellm
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        model_name = self.model
        if self.litellm_base_url and not model_name.startswith(("openai/", "azure/", "gemini/")):
            model_name = f"openai/{model_name}"
        
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "api_base": self.litellm_base_url
        }
        
        if self.api_key:
            kwargs["api_key"] = self.api_key
        
        response = litellm.completion(**kwargs)
        
        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        return response.choices[0].message.content
    
    def _generate_genai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        model_name = self.model.split("/")[-1]
        
        prompt_hash = hashlib.md5((system_prompt or '').encode('utf-8')).hexdigest()
        cache_key = f"{model_name}:{prompt_hash}"
        
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = self._genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
        model = self._model_cache[cache_key]
        
        generation_config = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            prompt_count = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_count = getattr(response.usage_metadata, 'candidates_token_count', 0)
            self.prompt_tokens += prompt_count
            self.completion_tokens += completion_count
        
        return response.text
    
    def warmup(self, system_prompt: Optional[str] = None) -> None:
        """Pre-warm the model and API connection."""
        if self._client_type != "genai":
            return
        
        model_name = self.model.split("/")[-1]
        cache_key = f"{model_name}:{hash(system_prompt or '')}"
        
        if cache_key not in self._model_cache:
            self._model_cache[cache_key] = self._genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
        
        model = self._model_cache[cache_key]
        try:
            response = model.generate_content(
                "hi",
                generation_config=self._genai.GenerationConfig(
                    temperature=0,
                    max_output_tokens=1
                )
            )
        except Exception as e:
            pass
    
    def _generate_requests(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using direct API requests."""
        import requests
        
        model_name = self.model.split("/")[-1]
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        content = prompt
        if system_prompt:
            content = f"{system_prompt}\n\n{prompt}"
        
        data = {
            "contents": [{"parts": [{"text": content}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        response = requests.post(
            url,
            json=data,
            params=params,
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        
        if 'usageMetadata' in result:
            usage = result['usageMetadata']
            prompt_count = usage.get('promptTokenCount', 0)
            completion_count = usage.get('candidatesTokenCount', 0)
            self.prompt_tokens += prompt_count
            self.completion_tokens += completion_count
        
        return result['candidates'][0]['content']['parts'][0]['text']
    
    def expand_query(self, query: str) -> str:
        """Expand a query with related concepts and synonyms."""
        expand_prompt = f"""Given the following query, expand it with related concepts, synonyms, and alternative phrasings that might help find relevant information in a knowledge graph.

Original query: {query}

Provide an expanded version that includes:
1. Key entities or concepts mentioned
2. Related terms and synonyms
3. Alternative phrasings

Keep the expansion concise (1-2 sentences max). Return ONLY the expanded query, nothing else.

Expanded query:"""
        
        try:
            expanded = self.generate(
                expand_prompt,
                temperature=0.3,
                max_tokens=200
            )
            return expanded.strip()
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query
    
    def generate_response(
        self,
        context: str,
        query: str,
        temperature: float = 0.7
    ) -> str:
        """Generate final response using context."""
        system_prompt = """You are a helpful assistant that answers questions based on provided knowledge graph context.
Be accurate, cite specific entities when relevant, and acknowledge if information is incomplete."""
        
        prompt = f"""## Knowledge Graph Context

{context}

## Question

{query}

## Answer

Based on the knowledge graph context above, provide a clear and accurate answer:"""
        
        return self.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "call_count": self.call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "model": self.model
        }
    
    def reset_stats(self):
        """Reset usage statistics."""
        self.call_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
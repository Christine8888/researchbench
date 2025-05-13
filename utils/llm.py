import os
import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
import ollama
import google.generativeai as genai
import google.api_core.exceptions

"""
LLM class
- initialize LLM backend from config file
- dummy backend for testing
- set system prompt, generate text
- (TODO) message buffer for multi-turn conversations, tool calling, etc.
"""

# CUSTOM CLASS FOR LLM
class LLM(ABC):
    """base class for language model clients"""
    def __init__(
        self, 
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.extra_params = kwargs
        self.client = None

        self.buffer = []
    
    @abstractmethod
    def initialize_client(self, **kwargs):
        """initialize specific provider client"""
        pass
    
    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """generate text using the API"""
        pass
    
    def set_system_prompt(self, system_prompt: str):
        """Update the system prompt"""
        self.system_prompt = system_prompt
    
    def add_to_buffer(self, messages: List[Dict[str, str]]):
        """Add messages to the message buffer"""
        self.buffer.extend(messages)
    
    def clear_buffer(self):
        """Clear the message buffer"""
        self.buffer = []

# GEMINI CLIENT
class GeminiClient(LLM):
    """Use Google Generative AI API for inference"""
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, system_prompt: str = ""):
        super().__init__(model_name, temperature, max_tokens, system_prompt)
        self.api_key = api_key
        self.initialize_client()
    
    def initialize_client(self):
        """Initialize client with API key"""
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(self.model_name)
    
    def generate_text(self, prompt: str) -> str:
        """Generating text using default API"""
        response = self.client.generate_content(self.system_prompt + "\n\n" + prompt)
        return response.text


class OLLAMAClient(LLM):
    """Use OLLAMA API for locally hosted inference"""
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, system_prompt: str = ""):
        super().__init__(model_name, temperature, max_tokens, system_prompt)
    
    def generate_text(self, prompt: str) -> str:
        return ollama.generate(model=self.model_name, prompt=self.system_prompt + "\n\n" + prompt)


class OpenAIClient(LLM):
    """OpenAI API client implementation"""
    
    def __init__(
        self, 
        api_key: str, 
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, system_prompt, **kwargs)
        self.api_key = api_key
        self.initialize_client()
    
    def initialize_client(self, **kwargs):
        self.client = OpenAI(api_key=self.api_key)
    
    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            **self.extra_params
        )
        return response.choices[0].message.content


class AzureOpenAIClient(LLM):
    """Azure OpenAI API client implementation"""
    
    def __init__(
        self, 
        api_key: str,
        endpoint: str,
        api_version: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, system_prompt, **kwargs)
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.initialize_client()
    
    def initialize_client(self, **kwargs):
        self.client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )
    
    def generate_text(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": "SYSTEM PROMPT: " + self.system_prompt + "\n\nUSER PROMPT: " + prompt},
            ],
            temperature=self.temperature,
            **self.extra_params
        )
        return response.choices[0].message.content


class AnthropicClient(LLM):
    """Anthropic API client implementation"""
    
    def __init__(
        self, 
        api_key: str,
        model_name: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str = "",
        **kwargs
    ):
        super().__init__(model_name, temperature, max_tokens, system_prompt, **kwargs)
        self.api_key = api_key
        self.initialize_client()
    
    def initialize_client(self, **kwargs):
        self.client = Anthropic(api_key=self.api_key)
    
    def generate_text(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            **self.extra_params
        )
        return response.content[0].text

class NotLLM(LLM):
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1024, system_prompt: str = ""):
        super().__init__(model_name, temperature, max_tokens, system_prompt)
    
    def generate_text(self, prompt: str) -> str:
        return "test string"

class ClientFactory:
    """create LLM clients from provider"""
    
    @staticmethod
    def create_client(provider: str, **config) -> LLM:
        """
        create and return the appropriate LLM client based on provider
        
        Args:
            provider: LLM provider name (openai, azure, anthropic, together)
            **config: configuration parameters for the client
            
        Returns:
            initialized LLM client
        """
        provider = provider.lower()
        
        if provider == "openai":
            return OpenAIClient(**config)
        elif provider == "azure":
            return AzureOpenAIClient(**config)
        elif provider == "anthropic":
            return AnthropicClient(**config)
        elif provider == "gemini":
            return GeminiClient(**config)
        elif provider == "not":
            return NotLLM()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @staticmethod
    def from_config(config_path: str) -> LLM:
        """
        create an LLM client from a configuration file
        
        Args:
            config_path: path to the YAML/JSON configuration file
            
        Returns:
            initialized LLM client
        """
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # load configuration
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # extract provider and create client
        provider = config.pop("provider", None)
        if not provider:
            raise ValueError("Provider not specified in config file")
        
        return ClientFactory.create_client(provider, **config)

if __name__ == "__main__":
    client = ClientFactory.from_config("config.yaml")
    print(client.generate_text("Hello, world!"))
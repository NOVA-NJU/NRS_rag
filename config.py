import os
from typing import Dict, Any

class RAGConfig:
    # 向量服务配置
    VECTOR_SERVICE_URL = os.getenv("VECTOR_SERVICE_URL", "http://localhost:8000")
    VECTOR_SEARCH_ENDPOINT = "/vectors/search"
    
    # LLM配置
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    
    # 不再需要Embedding模型配置
    
    # RAG参数
    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    
    # 提示词模板
    PROMPT_TEMPLATE = """请根据以下上下文信息回答问题。如果上下文中有相关信息，请基于这些信息回答；如果没有足够信息，请说明信息不足。

问题：{question}

相关上下文：
{context}

请基于以上上下文提供准确、有用的回答："""
    
    # API配置
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8003))

config = RAGConfig()
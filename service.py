import json
import requests
import numpy as np
from typing import List, Dict, Any
import logging
import os
import sys
from model import SourceDocument, AnswerResponse
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # 注意：向量服务的URL和端点需要根据实际情况调整
        self.vector_service_url = "http://localhost:8000/vectors/search"
    
    async def search_vector_db(self, question: str, top_k: int = None) -> List[Dict]:
        """调用向量数据库服务进行文本搜索"""
        if top_k is None:
            top_k = config.TOP_K
            
        try:
            # 根据实际向量服务API调整payload
            payload = {
                "query": question,  # 根据实际API使用"query"而不是"query_text"
                "top_k": top_k
            }
            
            logger.info(f"调用向量服务: {self.vector_service_url}")
            logger.info(f"请求参数: {payload}")
            
            response = requests.post(
                self.vector_service_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            logger.info(f"向量服务响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"向量服务返回: {len(result.get('results', []))} 个结果")
                
                # 转换格式以适应RAG服务期望的结构
                formatted_results = self._format_vector_results(result.get('results', []))
                return formatted_results
            else:
                logger.error(f"Vector service HTTP error: {response.status_code}, Response: {response.text}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Vector service request failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _format_vector_results(self, raw_results: List[Dict]) -> List[Dict]:
        """将向量服务的结果格式转换为RAG服务期望的格式"""
        formatted = []
        
        for result in raw_results:
            # 从metadata或其他字段中提取文本信息
            text_content = self._extract_text_content(result)
            source_info = self._extract_source_info(result)
            
            formatted_result = {
                "text": text_content,
                "source": source_info,
                "title": result.get('document_id', '未知文档'),
                "score": result.get('score', 0.0)
            }
            formatted.append(formatted_result)
            
        logger.info(f"格式化后得到 {len(formatted)} 个结果")
        return formatted
    
    def _extract_text_content(self, result: Dict) -> str:
        """从结果中提取文本内容"""
        # 尝试从不同字段提取文本内容
        if 'text' in result:
            return result['text']
        elif 'content' in result:
            return result['content']
        elif 'document_id' in result:
            # 如果没有具体文本，使用document_id作为占位符
            return f"文档: {result['document_id']}"
        else:
            return "暂无具体内容"
    
    def _extract_source_info(self, result: Dict) -> str:
        """从结果中提取来源信息"""
        metadata = result.get('metadata', {})
        if isinstance(metadata, dict):
            source = metadata.get('source', '')
            category = metadata.get('category', '')
            if source and category:
                return f"{category} - {source}"
            elif source:
                return source
            elif category:
                return category
        
        return result.get('document_id', '未知来源')
    
    async def call_llm(self, prompt: str) -> str:
        """调用大语言模型生成答案"""
        try:
            if config.LLM_PROVIDER == "ollama":
                return await self._call_ollama(prompt)
            elif config.LLM_PROVIDER == "openai":
                return await self._call_openai(prompt)
            else:
                raise ValueError(f"Unsupported LLM provider: {config.LLM_PROVIDER}")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    async def _call_ollama(self, prompt: str) -> str:
        """调用本地Ollama模型"""
        try:
            url = f"{config.OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            }
            
            logger.info(f"调用Ollama: {url}")
            
            response = requests.post(url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("response", "").strip()
                logger.info(f"Ollama返回答案长度: {len(answer)}")
                return answer
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                raise Exception(f"Ollama API returned status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            raise
    
    async def _call_openai(self, prompt: str) -> str:
        """调用OpenAI API"""
        try:
            import openai
            openai.api_key = config.OPENAI_API_KEY
            
            response = await openai.ChatCompletion.acreate(
                model=config.OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}],
                timeout=60
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI call failed: {e}")
            raise
    
    def build_prompt(self, question: str, context_docs: List[Dict]) -> str:
        """构建LLM提示词"""
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            context_parts.append(f"[{i}] {doc['text']}")
        
        context = "\n\n".join(context_parts) if context_parts else "暂无相关上下文信息"
        
        prompt = config.PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )
        
        logger.info(f"构建的提示词长度: {len(prompt)}")
        return prompt
    
    def format_sources(self, vector_results: List[Dict]) -> List[SourceDocument]:
        """格式化来源信息"""
        sources = []
        for result in vector_results:
            # 过滤低相似度的结果
            if result.get("score", 0) < config.SIMILARITY_THRESHOLD:
                continue
                
            source_doc = SourceDocument(
                text=result["text"],
                url=result.get("source", ""),
                title=result.get("title", "未知标题"),
                score=result.get("score")
            )
            sources.append(source_doc)
        
        logger.info(f"格式化后来源数量: {len(sources)}")
        return sources
    
    async def generate_answer(self, question: str) -> AnswerResponse:
        """完整的RAG流程"""
        logger.info(f"开始处理问题: {question}")
        
        # 1. 调用向量数据库进行文本搜索
        vector_results = await self.search_vector_db(question)
        logger.info(f"从向量数据库检索到 {len(vector_results)} 个文档")
        
        if not vector_results:
            logger.warning("没有检索到相关文档")
            return AnswerResponse(
                code="404",
                answer="抱歉，没有找到相关的信息来回答这个问题。",
                sources=[]
            )
        
        # 2. 构建提示词并调用LLM
        prompt = self.build_prompt(question, vector_results)
        logger.info("提示词构建完成")
        
        answer = await self.call_llm(prompt)
        logger.info("LLM生成答案完成")
        
        # 3. 格式化来源信息
        sources = self.format_sources(vector_results)
        logger.info(f"格式化 {len(sources)} 个来源")
        
        return AnswerResponse(
            answer=answer,
            sources=sources
        )

# 创建全局服务实例
rag_service = RAGService()
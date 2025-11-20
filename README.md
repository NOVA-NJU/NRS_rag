# 南京大学 RAG 问答系统

基于 RAG (Retrieval-Augmented Generation) 技术的南京大学信息问答系统，能够智能回答关于南京大学各部门、学院、通知公告等相关问题。

## 系统架构

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   前端界面       │    │   RAG 服务       │    │   向量数据库     │
│                 │    │   (端口: 8003)   │    │   (端口: 8000)   │
│   Web/Mobile    │◄──►│   FastAPI        │◄──►│   ChromaDB      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │   LLM 服务       │
                       │   (Ollama)       │
                       │   (端口: 11434)  │
                       │                  │
                       └──────────────────┘
```

## 功能特性

- 🔍 **智能检索**: 基于向量相似度搜索相关文档片段
- 🤖 **智能回答**: 使用大语言模型生成准确、自然的回答
- 📚 **来源追溯**: 提供答案的来源文档和可信度评分
- 🚀 **高性能**: 基于 FastAPI 的异步高性能 API
- 🔧 **易扩展**: 模块化设计，支持多种 LLM 和向量数据库

## 快速开始

### 环境要求

- Python 3.8+
- Ollama (用于本地 LLM)
- 向量数据库服务 (运行在端口 8000)

### 安装步骤

1. **克隆项目**
```bash
git clone <项目地址>
cd NRS_rag
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
创建 `.env` 文件：
```env
# RAG服务配置
API_HOST=0.0.0.0
API_PORT=8003

# 向量服务配置
VECTOR_SERVICE_URL=http://localhost:8000

# LLM配置
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# RAG参数
TOP_K=3
SIMILARITY_THRESHOLD=0.7
```

4. **启动服务**
```bash
# 启动 RAG 服务
python main.py
```

服务将在 http://localhost:8003 启动

### 验证安装

访问 API 文档：http://localhost:8003/docs

测试问答接口：
```bash
curl -X POST "http://localhost:8003/api/rag/" \
     -H "Content-Type: application/json" \
     -d '{"question": "南京大学图书馆开放时间是多少？"}'
```

## API 接口

### 问答接口

**URL**: `POST /api/rag/`

**请求示例**:
```json
{
  "question": "南京大学计算机系的研究生招生条件是什么？"
}
```

**响应示例**:
```json
{
  "code": "200",
  "answer": "根据相关资料，南京大学计算机系研究生招生需要...",
  "sources": [
    {
      "text": "研究生需具备计算机相关专业本科学历...",
      "url": "https://cs.nju.edu.cn/admission/2024",
      "title": "2024年研究生招生简章",
      "score": 0.95
    }
  ]
}
```

### 健康检查

**URL**: `GET /health`

**响应**:
```json
{
  "code": "200",
  "status": "healthy",
  "service": "RAG API"
}
```

## 配置说明

### 主要配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `API_PORT` | RAG 服务端口 | 8003 |
| `VECTOR_SERVICE_URL` | 向量服务地址 | http://localhost:8000 |
| `LLM_PROVIDER` | LLM 服务提供商 | ollama |
| `OLLAMA_MODEL` | 使用的模型 | qwen3:8b |
| `TOP_K` | 检索文档数量 | 3 |
| `SIMILARITY_THRESHOLD` | 相似度阈值 | 0.7 |

### 支持的 LLM 提供商

- **Ollama** (推荐): 本地部署，免费使用
- **OpenAI**: 云端 API，需要 API Key

## 项目结构

```
rag/
├── main.py              # 服务入口
├── config.py            # 配置管理
├── router.py            # API 路由
├── service.py          # RAG 核心逻辑
├── model.py            # 数据模型
├── requirements.txt     # 依赖列表
├── .env                 # 环境配置
└── README.md           # 项目说明
```

## 开发指南

### 添加新的 LLM 提供商

1. 在 `services.py` 的 `RAGService` 类中添加新的调用方法
2. 在 `call_llm` 方法中添加新的提供商分支
3. 更新 `config.py` 添加相应的配置参数

### 自定义提示词模板

修改 `config.py` 中的 `PROMPT_TEMPLATE`：

```python
PROMPT_TEMPLATE = """根据以下上下文回答问题：

问题：{question}

上下文：
{context}

请基于上下文提供准确的回答："""
```

### 调整检索参数

- **TOP_K**: 控制检索文档数量，值越大召回率越高但可能引入噪声
- **SIMILARITY_THRESHOLD**: 控制相似度阈值，值越高要求越严格

## 故障排除

### 常见问题

1. **向量服务连接失败**
   - 检查向量服务是否运行在端口 8000
   - 确认 `VECTOR_SERVICE_URL` 配置正确

2. **Ollama 服务未响应**
   - 确认 Ollama 已安装并运行
   - 检查 `OLLAMA_BASE_URL` 和 `OLLAMA_MODEL` 配置

3. **端口被占用**
   - 修改 `API_PORT` 使用其他端口
   - 或停止占用端口的进程

### 日志查看

服务启动时会显示详细的日志信息，包括：
- 模型加载状态
- API 请求处理
- 错误和警告信息

## 性能优化建议

1. **批量处理**: 对于大量查询，可以实现批量处理接口
2. **缓存机制**: 对常见问题的答案进行缓存
3. **异步处理**: 利用 FastAPI 的异步特性提高并发性能
4. **模型优化**: 根据需求选择合适的模型大小

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

## 许可证

本项目基于 MIT 许可证开源。

## 联系方式

如有问题，请通过以下方式联系：
- 创建 GitHub Issue
- 发送邮件至项目维护者

---

**注意**: 在使用本系统前，请确保已获得相关数据的使用权限，并遵守南京大学的相关规定和法律法规。
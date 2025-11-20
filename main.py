from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from router import router
from config import config

# 创建FastAPI应用
app = FastAPI(
    title="南京大学RAG问答系统",
    description="基于RAG的南京大学信息问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 生产环境中应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)

@app.get("/")
async def root():
    """根端点"""
    return {
        "code": "200",
        "message": "南京大学RAG问答系统API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,  # 开发时启用热重载
        log_level="info"
    )
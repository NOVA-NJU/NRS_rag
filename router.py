from fastapi import APIRouter, HTTPException, status
from model import QuestionRequest, AnswerResponse, ErrorResponse
from service import rag_service
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/api/rag/",
    response_model=AnswerResponse,
    summary="RAG问答接口",
    description="接收用户问题,通过RAG流程生成答案并返回相关来源"
)
async def rag_question(request: QuestionRequest):
    """
    RAG问答接口
    
    - **question**: 用户问题
    """
    try:
        if not request.question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="问题不能为空"
            )
        
        response = await rag_service.generate_answer(request.question)
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG process failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理问题时发生错误: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "code": "200",
        "status": "healthy",
        "service": "RAG API"
    }
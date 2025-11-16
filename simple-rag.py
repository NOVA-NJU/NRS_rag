from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from langchain_core.documents import Document

# 初始化通义千问大模型
llm = ChatTongyi(
    model="qwen-max",
    api_key=""
)

from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v1",
    dashscope_api_key=""
)
print("✓ 使用阿里云嵌入模型")

# 文档加载部分
try:
    loader = WebBaseLoader(
        "http://www.nju.edu.cn/info/1055/445831.htm",
        requests_kwargs={
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        }
    )
    docs = loader.load()
except Exception as e:
    print(f"WebBaseLoader 失败: {e}")
    try:
        response = requests.get(
            "http://www.nju.edu.cn/info/1055/445831.htm",
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=30
        )
        response.encoding = 'utf-8'
        docs = [Document(
            page_content=response.text,
            metadata={"source": "http://www.nju.edu.cn/info/1055/445831.htm"}
        )]
        print("使用 requests 成功获取内容")
    except Exception as e2:
        print(f"requests 也失败: {e2}")
        docs = [Document(
            page_content="""
            南京大学是一所历史悠久、声誉卓著的百年名校。学校现有四个校区，33个院系，本科生13934人、硕士研究生18158人、博士研究生8948人、留学生1691人。
            学校拥有一支高素质的师资队伍，其中包括中国科学院院士29人，中国工程院院士5人。
            南京大学在教学、科研和社会服务等方面取得了一系列重要成果，为国家培养了大批优秀人才。
            """,
            metadata={"source": "local_test_data"}
        )]

# 打印文档信息
for doc in docs:
    print("URL:", doc.metadata["source"])
    print("Content preview:", doc.page_content[:500])
    print("-" * 50)

# 继续后续的 RAG 流程
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
documents = text_splitter.split_documents(docs)
print(f"分割为 {len(documents)} 个文档块")

vector = FAISS.from_documents(documents, embeddings)

# 创建提示模板
prompt = ChatPromptTemplate.from_template("""仅根据提供的上下文回答问题：

{context}

Question: {input}""")

# 替代方案：如果 create_stuff_documents_chain 不可用
try:
    from langchain.chains.combine_documents import create_stuff_documents_chain

    document_chain = create_stuff_documents_chain(llm, prompt)
except ImportError:
    # 手动实现类似的链
    from langchain_core.runnables import RunnablePassthrough


    def simple_document_chain(input_dict):
        context = input_dict.get("context", "")
        question = input_dict.get("input", "")
        full_prompt = f"仅根据提供的上下文回答问题：\n\n{context}\n\nQuestion: {question}"
        return llm.invoke(full_prompt)


    document_chain = simple_document_chain

chain = document_chain | StrOutputParser()

retriever = vector.as_retriever(search_kwargs={"k": 3})

# 测试问题
test_questions = [
    "南京大学有哪些成就？",
    "南京大学的师资力量如何？"
]

for question in test_questions:
    print(f"\n问题: {question}")
    relevant_docs = retriever.invoke(question)
    answer = chain.invoke({
        "input": question,
        "context": relevant_docs
    })
    print(f"回答: {answer}")
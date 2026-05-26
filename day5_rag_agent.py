import os
import sys
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List


# ==========================================
# 1. 阿里云 DashScope Embedding 嵌入类
# ==========================================
class DashScopeEmbedding(Embeddings):
    """
    通过阿里云百炼 DashScope API，把文本转成向量。
    完全兼容 OpenAI 接口规范，只需要调整 base_url 和 api_key。
    """

    def __init__(self, client: OpenAI, model: str = "text-embedding-v4"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化（构建知识库时用）"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """单个查询向量化（用户提问时用）"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding


# ==========================================
# 2. 初始化客户端
# ==========================================
# 阿里云百炼使用兼容 OpenAI 的 base_url
dashscope_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# DeepSeek 客户端（对话用，不动）
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 创建嵌入模型实例
embeddings = DashScopeEmbedding(client=dashscope_client, model="text-embedding-v4")


# ==========================================
# 3. 构建知识库
# ==========================================
def build_knowledge_base(file_path: str):
    """把本地文档变成可检索的知识库"""
    # 3.1 加载文档
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("目前只支持 .txt 和 .pdf 文件")

    documents = loader.load()
    print(f"📄 文档加载完成，共 {len(documents)} 页/段")

    # 3.2 文档切块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每块 500 个字符
        chunk_overlap=50,  # 相邻块重叠 50 个字符
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 文档切块完成，共 {len(chunks)} 块")

    # 3.3 向量化并存入 FAISS
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    vectorstore.save_local("./knowledge_db")
    print(f"🧠 知识库构建完成，已保存到 ./knowledge_db")
    return vectorstore


# ==========================================
# 4. 加载已有知识库
# ==========================================
def load_knowledge_base():
    """加载本地已有的知识库"""
    return FAISS.load_local(
        "./knowledge_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


# ==========================================
# 5. RAG 问答 Agent
# ==========================================
def rag_qa(vectorstore, user_question: str, top_k: int = 3):
    """基于知识库回答问题"""
    print(f"❓ 用户问题：{user_question}")

    # 5.1 检索
    relevant_docs = vectorstore.similarity_search(user_question, k=top_k)
    print(f"🔍 检索到 {len(relevant_docs)} 个相关段落")

    # 5.2 构造上下文
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 5.3 调用 DeepSeek 生成答案（对话部分保持不变）
    messages = [
        {
            "role": "system",
            "content": f"""你是一个知识库问答助手。请严格根据以下参考资料回答用户问题。
如果参考资料中没有答案，请如实说"资料中未找到相关信息"，不要自己瞎编。

参考资料：
{context}"""
        },
        {"role": "user", "content": user_question}
    ]

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
        stream=True
    )

    print("🤖 Agent 回答：")
    full_answer = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            full_answer += text
    print()
    return full_answer


# ==========================================
# 6. 测试
# ==========================================
if __name__ == "__main__":
    # 准备测试文档
    sample_text = """
人工智能（AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

机器学习是人工智能的一个子集，它使系统能够从数据中自动学习和改进，而无需进行明确的编程。

深度学习是机器学习的一个分支，它使用多层神经网络来模拟人脑的学习过程。

Agent 智能体是一种能够感知环境并采取行动以实现目标的自主系统。在大型语言模型（LLM）的背景下，Agent 通常指能够使用工具、做出决策并与外部世界交互的 AI 系统。

RAG（检索增强生成）是一种结合了信息检索和文本生成的技术，它能让 LLM 基于外部知识库回答问题，从而减少幻觉。
    """

    if not os.path.exists("test.txt"):
        with open("test.txt", "w", encoding="utf-8") as f:
            f.write(sample_text)
        print("📝 已创建测试文档 test.txt")

    file_path = "test.txt"

    # 构建或加载知识库
    if os.path.exists("./knowledge_db"):
        print("📂 发现已有知识库，直接加载...")
        vectorstore = load_knowledge_base()
    else:
        print("🔨 正在构建知识库...")
        vectorstore = build_knowledge_base(file_path)

    # 开始问答
    print("\n" + "=" * 50)
    print("RAG 知识库问答系统已就绪，输入 'quit' 退出")
    print("=" * 50 + "\n")
    while True:
        question = input("❓ 请输入你的问题：")
        if question.lower() == "quit":
            break
        rag_qa(vectorstore, question)
        print()
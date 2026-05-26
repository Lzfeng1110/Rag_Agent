import os
import sys
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from typing import List
import streamlit as st


# ==========================================
# 1. 阿里云嵌入类（同 Day 5，无改动）
# ==========================================
class DashScopeEmbedding(Embeddings):
    def __init__(self, client: OpenAI, model: str = "text-embedding-v4"):
        self.client = client
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


# ==========================================
# 2. 初始化客户端（同 Day 5）
# ==========================================
dashscope_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
embeddings = DashScopeEmbedding(client=dashscope_client)


# ==========================================
# 3. 构建知识库（同 Day 5，无改动）
# ==========================================
@st.cache_resource  # Streamlit 缓存：只构建一次，不会重复调用 API
def build_knowledge_base(file_path: str):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("只支持 .txt 和 .pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore


# ==========================================
# 4. 加载知识库（同 Day 5）
# ==========================================
@st.cache_resource
def load_knowledge_base():
    return FAISS.load_local(
        "./knowledge_db",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


# ==========================================
# 5. RAG 问答函数（改造：支持记忆）
# ==========================================
def rag_qa_with_memory(vectorstore, user_question: str, chat_history: list, top_k: int = 3):
    # 检索相关文档
    relevant_docs = vectorstore.similarity_search(user_question, k=top_k)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # 构建消息列表（包含历史对话 + 当前问题）
    messages = [
        {"role": "system", "content": f"""你是一个知识库问答助手。请根据参考资料和对话历史回答用户问题。
如果参考资料中没有答案，请如实说"资料中未找到相关信息"。

参考资料：
{context}"""}
    ]
    # 添加历史对话
    for msg in chat_history:
        messages.append(msg)
    # 添加当前问题
    messages.append({"role": "user", "content": user_question})

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0.0,
        stream=True
    )
    # 流式返回答案
    for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ==========================================
# 6. Streamlit 界面
# ==========================================
st.set_page_config(page_title="私有知识库问答", page_icon="🧠")
st.title("🧠 私有知识库问答 Agent")

# 侧边栏：上传文档 & 初始化
with st.sidebar:
    st.header("📂 知识库管理")
    uploaded_file = st.file_uploader("上传 .txt 或 .pdf 文档", type=["txt", "pdf"])
    if uploaded_file:
        # 保存上传文件
        with open(f"./{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"已上传 {uploaded_file.name}")
        with st.spinner("正在构建知识库，请稍候..."):
            vectorstore = build_knowledge_base(f"./{uploaded_file.name}")
        st.success("知识库构建完成！")
    else:
        if os.path.exists("./knowledge_db"):
            vectorstore = load_knowledge_base()
            st.info("已加载本地知识库")
        else:
            st.warning("请先上传文档，或确保本地存在 knowledge_db")
            vectorstore = None

# 初始化会话状态（记忆）
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # 存储所有对话历史

# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    if vectorstore is None:
        st.error("请先上传文档构建知识库！")
    else:
        # 显示用户消息
        with st.chat_message("user"):
            st.write(prompt)

        # 生成回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                streaming_response = rag_qa_with_memory(
                    vectorstore,
                    prompt,
                    st.session_state.chat_history
                )
                response_text = st.write_stream(streaming_response)

        # 更新对话历史
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
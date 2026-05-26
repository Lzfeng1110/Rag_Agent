import os
import json
import streamlit as st
from openai import OpenAI
from datetime import datetime

# ==========================================
# 1. 初始化 DeepSeek 客户端
# ==========================================
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# ==========================================
# 2. 工具函数：操作本地 JSON 文件
# ==========================================
TODO_FILE = "todos.json"

def _load_todos():
    """内部函数：从 JSON 文件加载待办列表"""
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_todos(todos):
    """内部函数：保存待办列表到 JSON 文件"""
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)

def add_todo(task: str, deadline: str = "未指定"):
    """添加一条待办事项"""
    todos = _load_todos()
    new_todo = {
        "id": len(todos) + 1,
        "task": task,
        "deadline": deadline,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    todos.append(new_todo)
    _save_todos(todos)
    return f"✅ 已添加待办：'{task}'，截止日期：{deadline}"

def query_todos():
    """查询所有待办事项"""
    todos = _load_todos()
    if not todos:
        return "📭 当前没有待办事项。"
    result_lines = ["📋 当前待办列表："]
    for t in todos:
        result_lines.append(f"  [{t['id']}] {t['task']}（截止：{t['deadline']}）")
    return "\n".join(result_lines)
# ==========================================
# 3. 工具说明书
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "添加一条待办事项...",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "待办内容"},
                    "deadline": {"type": "string", "description": "截止日期..."}
                },
                "required": ["task", "deadline"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_todos",
            "description": "查询当前所有待办事项...",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]
# ==========================================
# 4. Agent 类（同 Day 3 架构）
# ==========================================
class TodoAgent:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "你是一个个人日程管家。你能帮用户添加待办事项、查询待办列表。如果用户要求添加或查询，必须调用对应工具获取真实数据，不要自己瞎编。"}
        ]

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=tools,
                temperature=0.0,
                stream=False
            )
            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                final_answer = assistant_message.content or ""
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            # 保存工具申请单
            self.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    } for tc in assistant_message.tool_calls
                ]
            })

            # 逐个执行工具
            for tc in assistant_message.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)

                if func_name == "add_todo":
                    result = add_todo(args.get("task"), args.get("deadline", "未指定"))
                elif func_name == "query_todos":
                    result = query_todos()
                else:
                    result = f"未知工具：{func_name}"

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

    def clear_memory(self):
        self.messages = [self.messages[0]]
# ==========================================
# 5. Streamlit 界面
# ==========================================
st.set_page_config(page_title="日程管家 Agent", page_icon="📅")
st.title("📅 个人日程管家 Agent")

# 初始化 Agent 和聊天历史
if "agent" not in st.session_state:
    st.session_state.agent = TodoAgent()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 侧边栏：待办列表
with st.sidebar:
    st.header("📋 待办列表")
    if st.button("🔄 刷新列表"):
        st.rerun()
    todos = _load_todos()
    if not todos:
        st.info("暂无待办事项")
    else:
        for t in todos:
            st.write(f"**{t['id']}.** {t['task']}")
            st.caption(f"⏰ {t['deadline']}")

# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
if prompt := st.chat_input("告诉我你想做什么..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("处理中..."):
            response_text = st.session_state.agent.chat(prompt)
        st.write(response_text)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append({"role": "assistant", "content": response_text})
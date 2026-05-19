import os
from openai import OpenAI
import json
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
def get_current_weather(city:str):
    """
    获取指定城市的天气信息(模拟数据，未来可接真实API）
    """
    weather_db = {
        "北京": "晴天，25度，微风",
        "上海": "多云，28度，潮湿",
        "东京": "小雨，18度",
    }
    return weather_db.get(city, f"{city}天气数据暂未收录")
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {"城市名称，例如:北京，上海，昆明"}
            },
            "required":["city"]
        }
    }
]

def run_agent_with_tools(user_query):
    """
        给 LLM 配备工具，让它能自主决定是否调用工具
    """
    messages = [
        {"role": "system",
         "content": "你是一个万能助手，可以帮用户查天气。如果用户问天气，你必须调用工具获取实时数据，不要自己瞎猜。"},
        {"role": "user", "content": user_query}
    ]

    print(f"用户：{user_query}")
    print("Agent 思考中...")
    # 第一次调用：LLM 看到用户问题和可用工具，决定下一步动作
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        tools=tools,
        temperature=0.0,
        stream=False
    )
    assistant_message = response.choices[0].message
    # 检查 LLM 是否想调用工具
    if assistant_message.tool_calls:
        #LLM 想用工具，我们执行它
        tool_call = assistant_message.tool_calls[0]



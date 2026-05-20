import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# ==========================================
# 1. 定义你的真实工具函数（今天只用模拟数据）
# ==========================================
def get_current_weather(city: str):
    """
    查询某个城市的实时天气（模拟数据，未来可接真实 API）
    """
    # 模拟天气数据
    weather_db = {
        "北京": "晴天，25度，微风",
        "上海": "多云，28度，潮湿",
        "昆明": "晴天，28度，小热",
    }
    return weather_db.get(city, f"{city}天气数据暂未收录")

# ==========================================
# 2. 用 JSON Schema 描述这个工具（LLM 才能看懂）
# ==========================================
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "查询指定城市的实时天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

# ==========================================
# 3. Agent 核心循环：带工具调用的对话函数
# ==========================================
def run_agent_with_tools(user_query):
    """
    给 LLM 配备工具，让它能自主决定是否调用工具
    """
    messages = [
        {"role": "system", "content": "你是一个万能助手，可以帮用户查天气。如果用户问天气，你必须调用工具获取实时数据，不要自己瞎猜。"},
        {"role": "user", "content": user_query}
    ]

    print(f"用户：{user_query}")
    print("Agent 思考中...")

    # 第一次调用：LLM 看到用户问题和可用工具，决定下一步动作
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,
        temperature=0.0,  # 天气查询要精确，所以用 0 度
        stream=False
    )

    assistant_message = response.choices[0].message

    # 检查 LLM 是否想调用工具
    if assistant_message.tool_calls:
        # LLM 想用工具，我们执行它
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Agent 决定调用工具：{function_name}，参数：{arguments}")

        if function_name == "get_current_weather":
            city = arguments.get("city")
            function_result = get_current_weather(city)
            print(f"工具返回结果：{function_result}")

            # 把 LLM 的“工具调用要求”和“工具执行结果”都追加到对话历史
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": function_result
            })

            # 第二次调用 LLM：把工具结果发给它，让它总结成人话
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=0.5,
                stream=True  # 第二次我们用流式，体验一下打字机效果
            )

            # 流式打印最终答案
            full_answer = ""
            for chunk in second_response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    print(text, end="", flush=True)
                    full_answer += text
            print()  # 换行
            return full_answer

    else:
        # LLM 没调用工具，直接输出答案
        if assistant_message.content:
            print(assistant_message.content)
            return assistant_message.content
        else:
            return "Agent 没有给出有效回复"

# ==========================================
# 4. 测试
# ==========================================
if __name__ == "__main__":
    print("===== 测试1：需要调用工具 =====")
    run_agent_with_tools("昆明今天天气怎么样？")
    print("\n")
    print("===== 测试2：不需要调用工具 =====")
    run_agent_with_tools("讲个笑话吧")





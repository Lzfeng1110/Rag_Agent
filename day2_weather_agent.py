import os
from openai import OpenAI
import json
import requests
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
def get_current_weather(city:str):
    """
    获取指定城市的天气信息(调用免费 wttr.in API)
    """
    try:
        # 1. 构造请求 URL
        url = f"https://wttr.in/{city}?format=j1"
        # 2. 发起网络请求，设置超时
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 如果网络错误，直接报错
        # 3. 解析 JSON
        data = response.json()

        # 4. 提取天气信息
        current = data['current_condition'][0]
        weather_desc = current['weatherDesc'][0]['value']
        temp_c = current['temp_C']
        feels_like = current['FeelsLikeC']
        humidity = current['humidity']

        # 5. 返回格式化的天气结果
        result = (f"{city}实时天气：{weather_desc}，温度{temp_c}°C，"
                  f"体感温度{feels_like}°C，湿度{humidity}%")
        return result

    except requests.exceptions.RequestException as e:
        return f"网络请求失败：{e}"
    except (KeyError, IndexError) as e:
        return f"未能解析{city}的天气数据，请检查城市名是否正确"

def calculate(expression: str):
    """
    安全地计算一个数学表达式，返回计算结果
    expression: 数学表达式字符串，例如 "3 + 5 * 2"
    """
    try:
        # 用 eval 执行表达式，但限制只能使用数学运算
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称,例如北京、上海、昆明"
                    }
                }
            },
            "required":["city"]
        }
    }
    ,
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "安全地计算一个数学表达式，返回计算结果",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式字符串，例如 \"3 + 5 * 2\""
                    }
                }
            },
            "required":["expression"]
        }
    }
]


def run_agent_with_tools(user_query):
    """
        给 LLM 配备工具，让它能自主决定是否调用工具
    """
    messages = [
        {"role": "system",
         "content": "你是一个万能助手，可以帮用户查天气、做计算。如果用户问天气，你必须调用工具获取实时数据，不要自己瞎猜。如果需要计算，请调用calculate工具。"},
        {"role": "user", "content": user_query}
    ]

    print(f"用户：{user_query}")
    print("Agent 思考中...")

    # 循环处理，支持多次工具调用
    while True:
        # 调用 LLM
        response = client.chat.completions.create(
            model="deepseek-v4-pro",
            messages=messages,
            tools=tools,
            temperature=0.0,
            stream=False
        )
        assistant_message = response.choices[0].message

        # 检查 LLM 是否想调用工具
        if not assistant_message.tool_calls:
            # 没有工具调用，直接返回内容
            if assistant_message.content:
                print(assistant_message.content)
                return assistant_message.content
            else:
                print("Agent没有给出有效回复")
                return None

        # LLM 想用工具，执行所有工具调用
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Agent尝试调用工具：{function_name}，参数：{arguments}")

            if function_name == "get_current_weather":
                city = arguments.get("city")
                function_result = get_current_weather(city)
                print(f"工具返回结果：{function_result}")

            elif function_name == "calculate":
                expression = arguments.get("expression")
                function_result = calculate(expression)
                print(f"工具返回结果：{function_result}")
            else:
                function_result = f"未知工具：{function_name}"

            # 构建助手消息，包含 reasoning_content（如果存在）
            assistant_msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            }

            # 如果有 reasoning_content，需要添加到消息中
            if hasattr(assistant_message, 'reasoning_content') and assistant_message.reasoning_content:
                assistant_msg["reasoning_content"] = assistant_message.reasoning_content

            messages.append(assistant_msg)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": function_result
                })
            # # 第二次调用 LLM：把工具结果发给它，让它总结成人话
            # second_response = client.chat.completions.create(
            #     model="deepseek-chat",
            #     messages=messages,
            #     temperature=0.0,
            #     stream=True  # 第二次我们用流式，体验一下打字机效果
            # )
            # # 流式打印最终答案
            # full_answer = ""
            # for chunk in second_response:
            #     if chunk.choices[0].delta.content:
            #         text = chunk.choices[0].delta.content
            #         print(text, end="", flush=True)
            #         full_answer += text
            # print()  # 换行
            # return full_answer

#测试
if __name__ == "__main__":
    print("===== 测试1：查天气（真实API） =====")
    run_agent_with_tools("现在昆明市五华区天气怎么样？")
    print("\n")

    print("===== 测试2：计算器 =====")
    run_agent_with_tools("帮我算一下 128 乘以 3.14 加上 256")
    print("\n")

    print("===== 测试3：同时需要两个工具（LLM会选择） =====")
    run_agent_with_tools("昆明市五华区现在的温度是多少？然后帮我把这个温度换算成华氏度是多少？")








import os
import json
import requests
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


# ==========================================
# 工具函数（同 Day 2）
# ==========================================
def get_current_weather(city: str):
    """查询指定城市的实时天气（调用免费 wttr.in API）"""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        current = data['current_condition'][0]
        desc = current['weatherDesc'][0]['value']
        temp = current['temp_C']
        feels = current['FeelsLikeC']
        humidity = current['humidity']
        return f"{city}天气：{desc}，温度{temp}°C，体感{feels}°C，湿度{humidity}%"
    except Exception as e:
        return f"天气查询失败：{e}"


def calculate(expression: str):
    """安全计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算失败：{e}"


# ==========================================
# 工具说明书
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
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算一个数学表达式，如'3+5*2'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "需要计算的数学表达式"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# ==========================================
# 今天的核心：带记忆和多步推理的 Agent
# ==========================================
class TravelAgent:
    def __init__(self):
        """
        初始化 Agent。
        记忆的秘密：self.messages 在多次 chat() 调用之间一直存在。
        """
        self.messages = [
            {"role": "system",
             "content": "你是一个旅行规划助手。你能记住用户说过的偏好和目的地。如果用户问天气，必须调用工具查实时数据；如果要算预算，必须调用工具计算。当所有工具结果都拿到后，再给用户一个完整的回答。"}
        ]

    def chat(self, user_input: str) -> str:
        """
        处理用户输入，支持记忆和多步工具调用。
        """
        # 1. 把用户新消息加入记忆
        self.messages.append({"role": "user", "content": user_input})
        print(f"🧑 用户：{user_input}")

        # 2. 多步工具调用循环
        while True:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                tools=tools,
                temperature=0.0,
                stream=False
            )
            assistant_message = response.choices[0].message

            # 3. 如果 LLM 不再要求工具，输出最终回答
            if not assistant_message.tool_calls:
                final_answer = assistant_message.content or ""
                print(f"🤖 助手：{final_answer}")
                # 把最终回答也存入记忆
                self.messages.append({"role": "assistant", "content": final_answer})
                return final_answer

            # 4. LLM 要求工具，先保存它的申请单
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

            # 5. 逐个执行工具，并将结果追加为 tool 消息
            for tc in assistant_message.tool_calls:
                func_name = tc.function.name
                args = json.loads(tc.function.arguments)
                print(f"🔧 调用工具：{func_name}({args})")

                if func_name == "get_current_weather":
                    result = get_current_weather(args.get("city"))
                elif func_name == "calculate":
                    result = calculate(args.get("expression"))
                else:
                    result = f"未知工具：{func_name}"

                print(f"📊 工具结果：{result}")

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
            # 循环继续，让 LLM 根据工具结果决定下一步

    def clear_memory(self):
        """重置记忆，只保留 system prompt"""
        self.messages = [self.messages[0]]
        print("🧹 记忆已清除")


# ==========================================
# 测试
# ==========================================
if __name__ == "__main__":
    agent = TravelAgent()

    print("===== 测试1：多轮对话（记忆） =====")
    agent.chat("我计划去昆明旅游")
    agent.chat("帮我查一下那边天气怎么样？")
    agent.chat("刚才我说要去哪个城市来着？")

    print("\n===== 测试2：多步推理（天气+计算） =====")
    agent.clear_memory()
    agent.chat("昆明现在的温度是多少？帮我把这个温度换算成华氏度")
    #
    print("\n===== 测试3：重置记忆 =====")
    agent.clear_memory()
    agent.chat("我刚才说了什么？")  # 应该不记得
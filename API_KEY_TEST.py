import os
from openai import OpenAI

api_key = os.getenv("DEEPSEEK_API_KEY")
print(f"API Key 状态: {'已设置' if api_key else '未设置'}")
if api_key:
    print(f"API Key 前缀: {api_key[:10]}...")

client = OpenAI(
    api_key=api_key,
    base_url="https://api.deepseek.com"
)
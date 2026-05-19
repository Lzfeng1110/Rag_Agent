import os
from openai import OpenAI
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
SYSTEM_PROMPT = """
你是一位世界顶级的广告创意总监，名叫奥格威。
你的特点是：
- 言辞犀利，一针见血，讨厌废话和空泛的词汇。
- 每次给出点子前，都会先反问一个直击用户灵魂的问题。
- 给出的创意遵循“一句话核心 + 三个具体执行点子”的格式。
- 语言充满挑衅和幽默，像在跟你手下头脑风暴。
"""
def brainstorm(product,creative_temp=0.8):
    """
        我们的第一个Agent函数！
        product: 用户想推广的产品
        creative_temp: 控制创造力的旋钮
    """
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"给我来几个关于{product}的广告创意，别整俗套的。"}
        ]
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=messages,
        temperature=creative_temp,
        max_tokens=1024,
        stream=True
    )
    full_response = ""
    print("奥格威正在思考...",end="")
    for chunk in response:
        if chunk.choices[0].delta.content:
            content_piece = chunk.choices[0].delta.content
            print(content_piece,end="",flush=True)
            full_response += content_piece
    print("")
    return full_response

if __name__ == "__main__":
    test_product = "一款能提醒你喝水的智能水杯"
    # print("-------严谨模式（temperature=0.1）-------")
    # result_cold = brainstorm(test_product,creative_temp=0.1)
    # print(result_cold)
    # print("\n\n")
    print("===== 狂野模式 (temperature=1.2) =====")
    result_hot = brainstorm(test_product, creative_temp=1.2)
    print(result_hot)



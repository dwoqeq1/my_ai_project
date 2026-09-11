from openai import OpenAI

from config import DASHSCOPE_BASE_URL, MODEL_NAME, require_api_key

# ★ Key 从 .env 读取，不再硬编码
client = OpenAI(
    api_key=require_api_key(),
    base_url=DASHSCOPE_BASE_URL,
)

messages = [
    {"role": "system", "content": "你是一个资深的Python后端工程师，回答要简练、直击痛点。"},
    {"role": "user", "content": "FastAPI和Flask有什么区别？"}
]

print("\n--- 开始流式输出 ---")

stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    stream=True   # 开启流式
)

for chunk in stream:
    # 大模型每次返回一个 chunk，其中 delta.content 可能为 None（比如结束标记）
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)

from openai import OpenAI

from config import DASHSCOPE_BASE_URL, MODEL_NAME, require_api_key

# 1. 初始化客户端（通义千问）
# ★ Key 不再写死在代码里，统一从 .env 读取（见 config.py 与 .env.example）
client = OpenAI(
    api_key=require_api_key(),          # 没配置会直接给出清晰报错，而不是跑到一半 401
    base_url=DASHSCOPE_BASE_URL,
)

# 2. 构造消息列表
messages = [
    {"role": "system", "content": "你是一个资深的Python后端工程师，回答要简练、直击痛点。"},
    {"role": "user", "content": "FastAPI和Flask有什么区别？"}
]

# 3. 发起请求（非流式）
response = client.chat.completions.create(
    model=MODEL_NAME,     # 模型名也走配置，方便切换 qwen-plus / qwen-max
    messages=messages,
    temperature=0.7,
    top_p=0.8,
    max_tokens=500
)

# 4. 打印结果
print("大模型回复：", response.choices[0].message.content)
print("消耗Token：", response.usage.total_tokens)

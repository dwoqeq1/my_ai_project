# day5_multi_tools.py (修好缩进 + 删除重复函数 + 密钥外置 + 工具容错)

import asyncio
import json
import os
import smtplib
from email.header import Header
from email.mime.text import MIMEText

from openai import AsyncOpenAI

from config import (
    DASHSCOPE_BASE_URL,
    MODEL_NAME,
    SMTP_HOST,
    SMTP_PORT,
    require_api_key,
    require_sender_email,
)
from safe_math import safe_calculate
from tool_utils import run_tool

# ================= 配置区 =================
# ★ 1. 发件邮箱和 SMTP 授权码从 .env 读取，不再写死在代码里
#      （授权码不是邮箱登录密码，在邮箱设置里单独生成）
# ★ 2. 你的阿里云百炼 API Key 同样从 .env 读取
BASE_URL = DASHSCOPE_BASE_URL
# ==========================================

# 初始化大模型客户端
client = AsyncOpenAI(api_key=require_api_key(), base_url=BASE_URL)


# ==========================================
# 1. 定义三个“本地工具”的具体实现
# ==========================================

# 工具1：查天气（模拟）
async def get_current_weather(location: str, unit: str = "celsius"):
    weather_db = {
        "北京": {"celsius": "25°C", "condition": "晴朗"},
        "上海": {"celsius": "28°C", "condition": "多云"},
        "深圳": {"celsius": "30°C", "condition": "阵雨"},
    }
    info = weather_db.get(location, {"celsius": "未知", "condition": "数据未覆盖"})
    return f"{location}天气：{info['condition']}，温度 {info.get(unit, info['celsius'])}"


# 工具2：发送真实邮件
async def send_email(recipient: str, subject: str, body: str):
    """
    ★ 这是一个有真实外部副作用的工具：会真的把邮件发出去。
    所以做了两层防护：
      1) 收件人必须是合法邮箱格式，避免模型幻觉出一个奇怪地址
      2) 未配置 .env 里的发件凭据时，直接返回错误说明，而不是登录失败抛栈
    """
    if not isinstance(recipient, str) or "@" not in recipient or "." not in recipient.split("@")[-1]:
        return f"ERROR: 收件人地址不合法：{recipient!r}"

    try:
        sender_email, auth_code = require_sender_email()
    except RuntimeError as e:
        return f"ERROR: {e}"

    try:
        # 构造邮件
        message = MIMEText(body, 'plain', 'utf-8')
        message['From'] = Header(sender_email)
        message['To'] = Header(recipient)
        message['Subject'] = Header(subject, 'utf-8')

        # 连接 SMTP 服务器（SSL），超时 15 秒避免卡死整个 Agent
        server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15)
        try:
            server.login(sender_email, auth_code)
            server.sendmail(sender_email, [recipient], message.as_string())
        finally:
            server.quit()

        return f"✅ 邮件已成功发送至 {recipient}，主题：{subject}"
    except smtplib.SMTPAuthenticationError:
        return "ERROR: 邮件发送失败 - SMTP 认证被拒，请检查 .env 里的 SENDER_AUTH_CODE 是否为授权码（不是登录密码）"
    except smtplib.SMTPException as e:
        return f"ERROR: 邮件发送失败 - SMTP 错误：{e}"
    except Exception as e:
        return f"ERROR: 邮件发送失败 - {type(e).__name__}：{e}"


# 工具3：数学计算
async def calculate(expression: str):
    """
    ★ 原来用 eval()，即使做了字符白名单，`10**10**10` 仍能打满 CPU/内存。
    现在换成基于 AST 的 safe_calculate，只放行四则运算，并对幂指数和结果大小设上限。
    """
    return safe_calculate(expression)


# ==========================================
# 2. 工具注册表（路由）
# ==========================================
TOOLS_MAP = {
    "get_current_weather": get_current_weather,
    "send_email": send_email,
    "calculate": calculate,
}

# ==========================================
# 3. 工具描述（JSON Schema，供大模型理解）
# ==========================================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名称，如：北京、上海"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位，默认为 celsius"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "向指定邮箱发送一封邮件",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "收件人邮箱地址"},
                    "subject": {"type": "string", "description": "邮件主题"},
                    "body": {"type": "string", "description": "邮件正文内容"},
                },
                "required": ["recipient", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行基础数学运算（加减乘除）",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，例如：'3 + 5 * 2'"},
                },
                "required": ["expression"],
            },
        },
    },
]


# ==========================================
# 4. 核心 Agent 引擎（无需修改）
# ==========================================
async def run_agent(user_query: str):
    print(f"\n👤 用户问：{user_query}")
    messages = [{"role": "user", "content": user_query}]

    # 第一轮：大模型决定是否调用工具
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    assistant_msg = response.choices[0].message

    if not assistant_msg.tool_calls:
        print(f"🤖 直接回答：{assistant_msg.content}")
        return assistant_msg.content

    # 第二轮：执行工具调用
    # ★ 用 tool_utils.run_tool 统一处理：参数解析失败 / 工具不存在 / 工具抛异常
    #   都不会再让整个请求崩掉，而是把错误信息作为 tool 结果回喂给模型。
    messages.append(assistant_msg)
    for tool_call in assistant_msg.tool_calls:
        result = await run_tool(tool_call, TOOLS_MAP)
        print(f"🔧 工具返回：{result}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })

    # 第三轮：生成最终回答
    final_response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content
    print(f"🤖 最终回答：{final_answer}")
    return final_answer


# ==========================================
# 5. 测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 多工具 Agent 演示...\n")

    # 这两个用例只读不写，可以放心跑
    asyncio.run(run_agent("上海今天天气如何？"))
    print("\n" + "=" * 60 + "\n")
    asyncio.run(run_agent("帮我算一下 (100 + 200) * 3 等于多少？"))

    # ★ 邮件用例会真的把邮件发出去（不可撤回的外部动作），
    #   所以默认不跑。确要测试时设置环境变量 ENABLE_EMAIL_DEMO=1，
    #   并把下面收件人换成你自己的邮箱。
    if os.environ.get("ENABLE_EMAIL_DEMO", "0") == "1":
        print("\n" + "=" * 60 + "\n")
        print("⚠️ 邮件演示已开启，将真实发送邮件")
        asyncio.run(run_agent(
            "给 test@example.com 发一封邮件，主题是'周报'，正文写'本周项目进展顺利'"
        ))
    else:
        print("\n" + "=" * 60)
        print("ℹ️ 邮件演示已跳过（会真实发信）。如需测试请设置 ENABLE_EMAIL_DEMO=1")

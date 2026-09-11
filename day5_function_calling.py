# day5_function_calling.py
import asyncio

from openai import AsyncOpenAI

from config import DASHSCOPE_BASE_URL, MODEL_NAME, require_api_key
from tool_utils import parse_tool_arguments

# ========== 配置区 ==========
# ★ Key 从 .env 读取，不再硬编码在代码里
BASE_URL = DASHSCOPE_BASE_URL
# ============================

client = AsyncOpenAI(api_key=require_api_key(), base_url=BASE_URL)


# 1. 定义一个“伪”天气函数（模拟真实API调用）
async def fake_weather_api(location: str, unit: str = "celsius"):
    """模拟查天气，实际项目中这里可以替换为 requests.get(真实天气接口)"""
    # 假装查了一下数据库
    weather_data = {
        "北京": {"celsius": "25°C", "fahrenheit": "77°F", "condition": "晴朗"},
        "上海": {"celsius": "28°C", "fahrenheit": "82°F", "condition": "多云"},
        "深圳": {"celsius": "30°C", "fahrenheit": "86°F", "condition": "阵雨"},
    }
    info = weather_data.get(location, {"celsius": "未知", "condition": "数据未覆盖"})
    return f"{location}天气：{info['condition']}，温度 {info.get(unit, info['celsius'])}"


# 2. 定义大模型需要的“工具描述”（OpenAI 标准格式）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的当前天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认为摄氏度",
                    },
                },
                "required": ["location"],
            },
        },
    }
]


# 3. 核心流程：用户提问 -> 模型决策 -> 执行函数 -> 返回最终答案
async def run_agent(user_query: str):
    print(f"\n👤 用户问：{user_query}")

    # ----- 第一回合：把问题发给大模型，并带上工具说明书 -----
    messages = [{"role": "user", "content": user_query}]

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,            # 给大模型看“说明书”
        tool_choice="auto",     # 让大模型自己决定要不要用工具
    )

    # 获取大模型的回复内容
    assistant_message = response.choices[0].message

    # ----- 判断：大模型是否想调用工具？ -----
    if assistant_message.tool_calls:
        # 大模型决定调用工具！
        tool_call = assistant_message.tool_calls[0]
        function_name = tool_call.function.name

        # ★ 解析模型生成的参数（例如 {"location": "北京"}）
        #   原来直接 json.loads()，模型输出空串或半截 JSON 就会抛异常把整个 Agent 打崩。
        #   现在失败时把错误当作工具结果喂回模型，让它自己纠正。
        try:
            function_args = parse_tool_arguments(tool_call.function.arguments)
        except ValueError as e:
            weather_result = f"ERROR: 参数解析失败 - {e}"
            function_args = {}
            print(f"⚠️ 参数解析失败：{e}")
        else:
            print(f"🧠 大模型决定调用工具：{function_name}，参数：{function_args}")

            # ----- 执行真正的本地函数（代替真实API） -----
            if function_name == "get_current_weather":
                weather_result = await fake_weather_api(
                    location=function_args.get("location"),
                    unit=function_args.get("unit", "celsius")
                )
            else:
                weather_result = f"ERROR: 未知工具 {function_name}"

        print(f"🔧 工具返回结果：{weather_result}")

        # ----- 第二回合：把工具执行结果喂回给大模型 -----
        # 把第一回合的回复追加到历史中
        messages.append(assistant_message)
        # 追加“工具执行结果”（注意 role 必须是 tool）
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,  # 必须关联 ID
            "content": weather_result
        })

        # 再次请求大模型，这次它会根据工具结果生成最终的自然语言回复
        final_response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
        )

        final_answer = final_response.choices[0].message.content
        print(f"🤖 最终回答：{final_answer}")
        return final_answer

    else:
        # 大模型觉得不需要调用工具，直接回复
        print(f"🤖 直接回答：{assistant_message.content}")
        return assistant_message.content


# 4. 启动运行
if __name__ == "__main__":
    print("🚀 启动 Function Calling 演示...")
    # 测试用例 1：问天气
    asyncio.run(run_agent("北京今天天气怎么样？"))

    print("\n" + "="*40)

    # 测试用例 2：问一个不相关的问题（验证它不会瞎调用工具）
    asyncio.run(run_agent("你好，你是谁？"))

# day6_plan_execute.py
import asyncio
import json

from openai import AsyncOpenAI

from config import DASHSCOPE_BASE_URL, MODEL_NAME, require_api_key
from safe_math import safe_calculate
from tool_utils import parse_plan_text, run_tool

# ===== 配置区 =====
# ★ Key 从 .env 读取，不再硬编码
BASE_URL = DASHSCOPE_BASE_URL
# =================

client = AsyncOpenAI(api_key=require_api_key(), base_url=BASE_URL)


# ==========================================
# 1. 定义“可用工具”（和之前一样，但为了演示，我们只保留两个）
# ==========================================
async def get_current_weather(location: str):
    """查天气"""
    weather_db = {
        "北京": "25°C 晴朗",
        "上海": "28°C 多云",
        "深圳": "30°C 阵雨",
    }
    return weather_db.get(location, f"ERROR: {location} 天气数据未覆盖")


async def calculate(expression: str):
    """数学计算（安全求值，替代 eval）"""
    return safe_calculate(expression)


# 工具注册表
TOOLS_MAP = {
    "get_current_weather": get_current_weather,
    "calculate": calculate,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定城市的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "城市名"},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学运算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"},
                },
                "required": ["expression"],
            },
        },
    },
]


# ==========================================
# 2. 核心：规划器（Planer）
# ==========================================
async def plan(user_query: str) -> list:
    """
    让大模型把用户问题拆解成 3~5 步计划。
    ★ 改为要求模型直接输出 JSON 数组，解析比“找'步骤'关键字”稳得多；
      万一模型没按 JSON 输出，parse_plan_text 还有按行解析和整段兜底两层退化。
    """
    system_prompt = """你是一个任务规划专家。请把用户的目标拆解为 3~5 个具体的、可执行的步骤。
步骤之间要有逻辑顺序，后一步依赖前一步的结果。
只输出一个 JSON 数组，每个元素是一句话的步骤描述，不要输出任何其它文字。
示例格式：["查询上海天气", "计算3天打车总费用", "汇总结论"]"""

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户目标：{user_query}"}
        ],
        temperature=0.3,
    )

    plan_text = response.choices[0].message.content
    print(f"📋 规划器生成的原始计划：\n{plan_text}")

    # ★ 健壮解析（JSON 优先，其次按行，最后整段兜底）
    steps = parse_plan_text(plan_text)

    print(f"✅ 解析后的步骤列表（共 {len(steps)} 步）：")
    for i, s in enumerate(steps, 1):
        print(f"  步骤{i}: {s}")

    return steps


# ==========================================
# 3. 核心：执行器（Executor）
# ==========================================
async def execute_step(step_desc: str, context: dict) -> str:
    """
    执行单一步骤：大模型根据步骤描述，决定调用哪个工具并返回结果。
    context 里存放之前步骤的结果，供后续步骤参考。
    """
    # 把历史上下文拼成提示
    context_str = "\n".join([f"之前步骤结果：{v}" for v in context.values()])
    system_prompt = f"""你是一个任务执行专家。当前已有的上下文信息：
{context_str}

请根据当前步骤的描述，调用合适的工具来完成它。如果不需要工具，直接给出答案。
当前步骤：{step_desc}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"请执行步骤：{step_desc}"}
    ]

    # 调用大模型，带上工具
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )

    assistant_msg = response.choices[0].message

    # 如果大模型决定调用工具
    if assistant_msg.tool_calls:
        results = []
        for tool_call in assistant_msg.tool_calls:
            # ★ 统一容错执行，参数解析失败/工具不存在/工具抛异常都不再崩
            result = await run_tool(tool_call, TOOLS_MAP)
            print(f"   🔧 工具结果：{result}")
            results.append(result)

        step_result = "\n".join(results)
        print(f"   📤 步骤结果：{step_result}")
        return step_result

    # 如果大模型直接回答（不需要工具）
    answer = assistant_msg.content
    print(f"   💬 直接回答：{answer}")
    return answer


# ==========================================
# 4. 核心：Plan-and-Execute 总控制器
# ==========================================
async def plan_and_execute(user_query: str):
    print(f"\n🎯 用户目标：{user_query}")

    # 阶段1：规划
    steps = await plan(user_query)

    # 阶段2：执行（顺序执行，每步结果存入 context）
    context = {}
    for i, step in enumerate(steps, 1):
        print(f"\n🔄 执行步骤 {i}/{len(steps)}：{step}")
        result = await execute_step(step, context)
        context[f"步骤{i}"] = result

    # 阶段3：汇总
    print("\n" + "="*60)
    print("📊 汇总所有步骤结果：")
    for key, val in context.items():
        print(f"  {key}: {val}")

    # 让大模型生成最终总结
    summary_prompt = f"""用户最初的目标是：{user_query}

以下是执行每个步骤得到的结果：
{json.dumps(context, ensure_ascii=False, indent=2)}

请根据这些结果，给用户一个完整、自然的最终回答。"""

    final_response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.5,
    )

    final_answer = final_response.choices[0].message.content
    print(f"\n🤖 最终回答：\n{final_answer}")
    return final_answer


# ==========================================
# 5. 测试入口
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 Plan-and-Execute 演示...\n")

    # 一个需要多步推理的复杂任务
    asyncio.run(plan_and_execute(
        "我想去上海玩，查一下上海天气，然后帮我算算如果待3天，每天打车花费50元，总共需要多少交通费？"
    ))

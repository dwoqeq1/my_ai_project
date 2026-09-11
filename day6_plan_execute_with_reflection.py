# day6_plan_execute_with_reflection.py
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
# 1. 工具定义（模拟脆弱性，用于演示反思）
# ==========================================
async def get_current_weather(location: str):
    """查天气（故意只支持三个城市，触发反思）"""
    weather_db = {
        "北京": "25°C 晴朗",
        "上海": "28°C 多云",
        "深圳": "30°C 阵雨",
    }
    # 如果查不到，返回错误信息，触发反思
    if location not in weather_db:
        return f"ERROR: 未找到 '{location}' 的天气数据。"
    return weather_db[location]

async def calculate(expression: str):
    """数学计算（安全求值，替代 eval）"""
    return safe_calculate(expression)

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
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学运算",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
        },
    },
]

# ==========================================
# 2. 规划器（Planer）
# ==========================================
async def plan(user_query: str) -> list:
    # ★ 要求输出 JSON 数组，比字符串关键字匹配稳得多
    system_prompt = """你是一个任务规划专家。把目标拆解为 3~5 个具体步骤。
只输出一个 JSON 数组，每个元素是一句话的步骤描述，不要输出任何其它文字。
示例格式：["查询广州天气", "计算 10*5", "汇总结论"]"""
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户目标：{user_query}"}
        ],
        temperature=0.3,
    )
    plan_text = response.choices[0].message.content
    # ★ 健壮解析：JSON 优先，其次按行，最后整段兜底
    steps = parse_plan_text(plan_text)

    print(f"📋 规划完成，共 {len(steps)} 步")
    for i, s in enumerate(steps, 1):
        print(f"  步骤{i}: {s}")
    return steps

# ==========================================
# 3. 执行器（Executor）- 执行单个动作
# ==========================================
async def execute_action(step_desc: str, context: str) -> str:
    """执行一步，返回工具执行结果"""
    system_prompt = f"当前上下文：{context}\n请根据步骤描述调用工具。步骤：{step_desc}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"执行：{step_desc}"}
    ]

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    assistant_msg = response.choices[0].message

    if assistant_msg.tool_calls:
        results = []
        for tool_call in assistant_msg.tool_calls:
            # ★ 统一容错执行：参数非法/工具缺失/工具抛异常都返回 ERROR 串，
            #   而不是让整个流程崩掉——ERROR 串正好能被反思器识别并重试。
            result = await run_tool(tool_call, TOOLS_MAP)
            print(f"   🔧 工具结果：{result}")
            results.append(result)
        return "\n".join(results)

    return assistant_msg.content or "执行完成，无具体输出。"

# ==========================================
# 4. ★★★ 核心升级：反思器（Reflector） ★★★
# ==========================================
async def reflect(step_desc: str, raw_result: str) -> dict:
    """
    检查执行结果是否成功。
    返回：{"status": "PASS" / "FAIL", "suggestion": "修正建议"}
    """
    check_prompt = f"""你是一个严格的质量检查员。
目标步骤：{step_desc}
执行结果：{raw_result}

请判断该结果是否成功实现了目标。
- 如果结果中明显包含错误信息（如 ERROR, 失败, 无法, 未找到），或者结果为空，请回复 FAIL。
- 如果结果合理、有效，请回复 PASS。

如果判定为 FAIL，请用一句简短的话给出修正建议（例如：“重新查询广州的天气”或“检查表达式格式”）。

输出格式（严格按此 JSON）：
{{"status": "PASS" 或 "FAIL", "suggestion": "修正建议"}}"""

    try:
        resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content
        # 提取 JSON（防止模型加废话）
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end] if start != -1 else content
        result = json.loads(json_str)

        # ★ 校验结构，防止模型返回 {"status": "pass"} 小写或缺字段导致后续判断失灵
        status = str(result.get("status", "")).strip().upper()
        if status not in ("PASS", "FAIL"):
            print(f"   ⚠️ 反思器返回了未知状态 {status!r}，按 PASS 处理")
            status = "PASS"
        return {"status": status, "suggestion": str(result.get("suggestion", "")).strip()}
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # 解析失败默认 PASS，避免死循环
        print(f"   ⚠️ 反思器解析异常，默认通过：{type(e).__name__}: {e}")
        return {"status": "PASS", "suggestion": ""}
    except Exception as e:
        # 网络/接口异常也不该让整个流程挂掉
        print(f"   ⚠️ 反思器调用失败，默认通过：{type(e).__name__}: {e}")
        return {"status": "PASS", "suggestion": ""}

# ==========================================
# 5. 带反思的执行流程（带重试）
# ==========================================
async def execute_step_with_reflection(step_desc: str, context: dict, max_retries: int = 2):
    """执行步骤，如果失败则根据反思建议重试"""
    # ★ 只把「步骤N: 结果」拼进上下文。
    #   原代码把重试提示写回 context 字典（context["_retry_hint"]），
    #   会一路带到最终汇总的 prompt 里，污染答案。现在改用局部变量。
    def build_context_str(extra_hint: str = "") -> str:
        parts = [f"{k}: {v}" for k, v in context.items() if not k.startswith("_")]
        if extra_hint:
            parts.append(f"上次失败原因及修正建议：{extra_hint}")
        return "\n".join(parts)

    context_str = build_context_str()
    raw_result = "执行失败（未产生结果）"

    for attempt in range(1, max_retries + 1):
        print(f"   🏃 尝试第 {attempt} 次执行...")

        # 执行动作
        raw_result = await execute_action(step_desc, context_str)
        print(f"   📤 原始结果：{raw_result[:100]}...")

        # 调用反思器
        reflection = await reflect(step_desc, raw_result)

        if reflection.get("status") == "PASS":
            print(f"   ✅ 反思通过！")
            return raw_result

        suggestion = reflection.get("suggestion") or "请重试"
        print(f"   ❌ 反思失败，建议：{suggestion}")

        if attempt < max_retries:
            # ★ 修正建议只进局部上下文，不写回 context 字典
            context_str = build_context_str(suggestion)
            print(f"   🔄 准备根据建议重试...")
        else:
            print(f"   🚫 达到最大重试次数，返回当前结果（含错误）")
            return raw_result

    return raw_result

# ==========================================
# 6. Plan-and-Execute 总控（带反思）
# ==========================================
async def plan_and_execute(user_query: str):
    print(f"\n🎯 用户目标：{user_query}")

    # 阶段1：规划
    steps = await plan(user_query)

    # 阶段2：带反思的执行
    context = {}
    for i, step in enumerate(steps, 1):
        print(f"\n🔄 执行步骤 {i}/{len(steps)}：{step}")
        result = await execute_step_with_reflection(step, context)
        context[f"步骤{i}"] = result

    # 阶段3：汇总
    print("\n" + "="*60)
    summary_prompt = f"""用户目标：{user_query}
执行结果摘要：{json.dumps(context, ensure_ascii=False, indent=2)}
请生成自然的最终回答。如果某个步骤的结果是 ERROR，请如实说明该步骤未能完成，不要编造数据。"""
    final = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.5,
    )
    print(f"\n🤖 最终回答：\n{final.choices[0].message.content}")
    return final.choices[0].message.content

# ==========================================
# 7. 测试入口（专为触发反思设计）
# ==========================================
if __name__ == "__main__":
    print("🚀 启动 带反思的 Plan-and-Execute...\n")
    # 故意问一个“广州”天气，我们的天气库没有广州，会触发反思重试
    asyncio.run(plan_and_execute("查一下广州天气，再算算 10*5 等于多少？"))

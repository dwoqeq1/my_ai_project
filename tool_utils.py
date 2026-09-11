# tool_utils.py
# Function Calling 的公共容错逻辑。
#
# 为什么需要它：
#   模型返回的 tool_call.function.arguments 是一段「字符串形式的 JSON」。
#   直接 json.loads() 有三个常见翻车场景：
#     1) 字符串为空或 None（模型决定不带参数调用）
#     2) 被截断的半截 JSON（输出超长、网络中断）
#     3) 解析出来是 list / str 而不是 dict（模型没按 schema 输出）
#   原代码没有 try，任何一种都会让整个请求 500，Agent 直接崩掉。
#   这里统一兜住，并把错误信息作为「工具执行结果」喂回模型，
#   让它有机会自己纠正参数——这也是反思器能识别的 ERROR 格式。
import json
from typing import Any, Callable


def parse_tool_arguments(raw: Any) -> dict:
    """
    把模型给的 arguments 安全解析成 dict。
    解析失败时抛 ValueError，由调用方转成工具错误信息。
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):  # 有些 SDK 版本已经解析好了
        return raw
    if not isinstance(raw, str):
        raise ValueError(f"arguments 类型异常：{type(raw).__name__}")

    text = raw.strip()
    if not text:
        return {}  # 无参数调用是合法的

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"参数不是合法 JSON（{e.msg}）：{text[:120]}") from e

    if not isinstance(parsed, dict):
        raise ValueError(f"参数应为 JSON 对象，实际是 {type(parsed).__name__}：{text[:120]}")
    return parsed


async def run_tool(tool_call, tools_map: dict[str, Callable]) -> str:
    """
    执行单个 tool_call，永远返回字符串，不抛异常。
    返回的错误串带 ERROR: 前缀，方便反思器判定为失败。
    """
    func_name = getattr(getattr(tool_call, "function", None), "name", "") or "<未知工具名>"

    tool_func = tools_map.get(func_name)
    if tool_func is None:
        return f"ERROR: 未找到工具 '{func_name}'，可用工具：{', '.join(tools_map) or '（无）'}"

    try:
        func_args = parse_tool_arguments(tool_call.function.arguments)
    except ValueError as e:
        return f"ERROR: 工具 '{func_name}' 参数解析失败 - {e}"

    try:
        result = tool_func(**func_args)
        if hasattr(result, "__await__"):  # 兼容同步/异步两种工具实现
            result = await result
    except TypeError as e:
        # 典型场景：模型少传或多传了参数
        return f"ERROR: 工具 '{func_name}' 参数不匹配 - {e}（收到：{func_args}）"
    except Exception as e:
        return f"ERROR: 工具 '{func_name}' 执行异常 - {type(e).__name__}: {e}"

    if result is None:
        return f"工具 '{func_name}' 执行完成，无返回值。"
    return result if isinstance(result, str) else str(result)


def parse_plan_text(plan_text: str) -> list[str]:
    """
    把规划器输出的文本解析成步骤列表。
    兼容三种格式，按可靠性依次尝试：
      1) JSON 数组：["步骤一", "步骤二"]  ← 推荐，最稳
      2) 带序号的行：步骤1: xxx / Step 1: xxx / 1. xxx / - xxx
      3) 兜底：整段文本作为单一步骤

    原来只靠 `"步骤" in line` 判断，模型输出 `1.` 或 `- ` 开头时
    会整段退化成一步，计划就白做了。
    """
    if not plan_text or not plan_text.strip():
        return []
    text = plan_text.strip()

    # --- 1) 先尝试 JSON（可能被 ```json 包裹）---
    json_text = text
    if json_text.startswith("```"):
        json_text = json_text.strip("`")
        if json_text.lower().startswith("json"):
            json_text = json_text[4:]
        json_text = json_text.strip()
    start, end = json_text.find("["), json_text.rfind("]")
    if start != -1 and end > start:
        try:
            data = json.loads(json_text[start:end + 1])
            if isinstance(data, list):
                steps = [str(x).strip() for x in data if str(x).strip()]
                if steps:
                    return steps
        except json.JSONDecodeError:
            pass  # 继续走按行解析

    # --- 2) 按行解析，剥掉各种序号前缀 ---
    import re
    prefix_pattern = re.compile(
        r"^\s*(?:步骤|Step|STEP|step)?\s*\d*\s*[.:：、)\-*)]*\s*"
    )
    steps = []
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        if not line:
            continue
        # 跳过明显的说明性行
        if line.startswith("以下是") or line.startswith("好的") or line.startswith("```"):
            continue
        cleaned = prefix_pattern.sub("", line, count=1).strip()
        if not cleaned:
            cleaned = line
        # 去掉可能残留的 markdown 加粗
        cleaned = cleaned.replace("**", "").strip()
        if cleaned:
            steps.append(cleaned)

    # --- 3) 兜底 ---
    return steps if steps else [text]

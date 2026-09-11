# test_tool_utils.py
# tool_utils 模块的回归测试：计划解析、参数解析、工具执行容错。
#
# 运行方式：
#   venv\Scripts\python test_tool_utils.py
#   venv\Scripts\python -m pytest test_tool_utils.py
"""
这三块是 Agent 最容易崩的地方：
  1) 规划器输出格式漂移 -> 计划退化成一步
  2) 模型给出半截 JSON 参数 -> json.loads 抛异常，整个请求 500
  3) 工具内部抛异常 -> 未捕获，流程中断
本文件用假工具桩把各种翻车场景锁死，保证它们都退化成可读的 ERROR 串。
"""
import asyncio

from tool_utils import parse_plan_text, parse_tool_arguments, run_tool


class _FakeFunc:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    """模拟 openai SDK 的 tool_call 对象，避免测试依赖真实网络请求。"""

    def __init__(self, name, arguments):
        self.function = _FakeFunc(name, arguments)
        self.id = "call_test"


# ---------- 计划解析用例：(模型输出, 期望步骤数) ----------
PLAN_CASES = [
    ('["查询上海天气", "计算3天打车费", "汇总结论"]', 3),   # 标准 JSON（推荐格式）
    ('```json\n["步骤一", "步骤二"]\n```', 2),              # 被 markdown 代码块包裹
    ("步骤1: 查询天气\n步骤2: 计算费用\n步骤3: 汇总", 3),    # 原格式，英文冒号
    ("步骤1：查询天气\n步骤2：计算费用", 2),                 # 中文冒号（原来会解析失败）
    ("Step 1: query weather\nStep 2: calc cost", 2),        # 英文步骤
    ("1. 查询天气\n2. 计算费用\n3. 汇总结论", 3),            # 纯数字序号（原来会退化成 1 步）
    ("- 查询天气\n- 计算费用", 2),                          # markdown 列表（原来会退化）
    ("好的，以下是计划：\n1. 查询天气\n2. 计算费用", 2),      # 带寒暄前缀
    ("**步骤1**: 查询天气\n**步骤2**: 计算费用", 2),         # 带 markdown 加粗
    ("", 0),                                                # 空输出
    ("   ", 0),                                             # 纯空白
    ("直接执行这个任务", 1),                                 # 无格式，整段兜底
]

# ---------- 参数解析用例 ----------
ARG_GOOD = [
    ('{"location": "北京"}', {"location": "北京"}),
    ("", {}),                       # 无参数调用是合法的
    (None, {}),
    ({"a": 1}, {"a": 1}),           # 某些 SDK 版本已解析成 dict
    ('  {"expression": "1+1"}  ', {"expression": "1+1"}),   # 带空白
]
ARG_BAD = [
    '{"location": ',        # 被截断的半截 JSON
    "not json at all",      # 完全不是 JSON
    "[1,2,3]",              # 是 JSON 但不是对象
    '"just a string"',      # 是 JSON 但不是对象
]


def test_parse_plan_text():
    """各种规划器输出格式都要能解析出正确步骤数，不能整段退化。"""
    for text, expect in PLAN_CASES:
        got = parse_plan_text(text)
        assert len(got) == expect, f"{text[:40]!r} 解析出 {len(got)} 步，期望 {expect} 步：{got}"
        assert all(isinstance(s, str) and s.strip() for s in got), f"步骤含空值：{got}"


def test_parse_tool_arguments_good():
    """合法参数（含空参数）必须正确解析成 dict。"""
    for raw, expect in ARG_GOOD:
        assert parse_tool_arguments(raw) == expect, f"{raw!r} 解析结果不符预期"


def test_parse_tool_arguments_bad():
    """非法参数必须抛 ValueError，由上层转成工具错误信息回喂模型。"""
    for raw in ARG_BAD:
        try:
            parse_tool_arguments(raw)
        except ValueError:
            continue
        raise AssertionError(f"{raw!r} 是非法参数，却没有抛 ValueError")


# ---------- 工具容错用例：(场景名, tool_call, 是否应返回 ERROR) ----------
async def _good_tool(location: str):
    return f"{location} 25°C 晴朗"


async def _boom_tool(x: str):
    raise RuntimeError("工具内部炸了")


def _sync_tool(n: int):
    return f"同步工具结果 {n}"


async def _none_tool():
    return None


_TEST_TOOLS = {
    "good": _good_tool,
    "boom": _boom_tool,
    "sync": _sync_tool,
    "none": _none_tool,
}

RUN_CASES = [
    ("正常调用", _FakeToolCall("good", '{"location": "北京"}'), False),
    ("工具不存在", _FakeToolCall("ghost", "{}"), True),
    ("参数是半截JSON", _FakeToolCall("good", '{"location": '), True),
    ("参数缺必填项", _FakeToolCall("good", "{}"), True),
    ("参数多了个字段", _FakeToolCall("good", '{"location":"北京","zzz":1}'), True),
    ("工具内部抛异常", _FakeToolCall("boom", '{"x":"1"}'), True),
    ("同步工具也能跑", _FakeToolCall("sync", '{"n": 5}'), False),
    ("工具返回None", _FakeToolCall("none", ""), False),
]


def test_run_tool_never_raises():
    """任何翻车场景都必须返回字符串，绝不向上抛异常打断 SSE 流。"""

    async def _run():
        for label, tc, should_error in RUN_CASES:
            result = await run_tool(tc, _TEST_TOOLS)
            assert isinstance(result, str), f"{label}: 返回值不是字符串"
            is_error = result.startswith("ERROR")
            assert is_error == should_error, f"{label}: 期望 error={should_error}，实际={is_error}，结果={result[:70]}"

    asyncio.run(_run())


def test_error_messages_are_actionable():
    """ERROR 串要带上工具名和原因，反思器与用户才能据此纠正。"""

    async def _run():
        result = await run_tool(_FakeToolCall("ghost", "{}"), _TEST_TOOLS)
        assert "ghost" in result and "good" in result, f"未知工具的错误信息应列出工具名与可用工具：{result}"

        result = await run_tool(_FakeToolCall("boom", '{"x":"1"}'), _TEST_TOOLS)
        assert "boom" in result and "RuntimeError" in result, f"工具异常应带工具名与异常类型：{result}"

    asyncio.run(_run())


if __name__ == "__main__":
    print("=== tool_utils 回归测试 ===")

    print(f"\n[1] parse_plan_text（{len(PLAN_CASES)} 例）")
    for text, expect in PLAN_CASES:
        got = parse_plan_text(text)
        flag = "PASS" if len(got) == expect else "FAIL"
        print(f"    {flag}  {text[:38]!r} -> {len(got)} 步（期望 {expect}）")

    print(f"\n[2] parse_tool_arguments 合法输入（{len(ARG_GOOD)} 例）")
    for raw, expect in ARG_GOOD:
        got = parse_tool_arguments(raw)
        print(f"    {'PASS' if got == expect else 'FAIL'}  {raw!r} -> {got}")

    print(f"\n[3] parse_tool_arguments 非法输入（{len(ARG_BAD)} 例，应全部 BLOCKED）")
    for raw in ARG_BAD:
        try:
            parse_tool_arguments(raw)
            print(f"    LEAKED!  {raw!r} 没有抛错")
        except ValueError as e:
            print(f"    BLOCKED  {raw!r} -> {str(e)[:50]}")

    print(f"\n[4] run_tool 容错（{len(RUN_CASES)} 例）")

    async def _demo():
        for label, tc, should_error in RUN_CASES:
            result = await run_tool(tc, _TEST_TOOLS)
            is_error = result.startswith("ERROR")
            flag = "PASS" if is_error == should_error else "FAIL"
            print(f"    {flag}  {label}: {result[:68]}")

    asyncio.run(_demo())

    test_parse_plan_text()
    test_parse_tool_arguments_good()
    test_parse_tool_arguments_bad()
    test_run_tool_never_raises()
    test_error_messages_are_actionable()
    total = len(PLAN_CASES) + len(ARG_GOOD) + len(ARG_BAD) + len(RUN_CASES)
    print(f"\n✅ 全部 {total} 项断言通过")

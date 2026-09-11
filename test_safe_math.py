# test_safe_math.py
# safe_math 模块的回归测试。
#
# 运行方式：
#   venv\Scripts\python test_safe_math.py        # 直接跑
#   venv\Scripts\python -m pytest test_safe_math.py   # 或用 pytest
"""
本项目的 calculate 工具原本用 eval()，即使做了字符白名单过滤，
`10**10**10` 这类合法表达式仍会耗尽 CPU/内存。
这些用例锁死「正常算式必须算对」和「危险输入必须被拒」两条底线，
后续任何人改 safe_math 都能立刻发现回归。
"""
from safe_math import MathEvalError, safe_calculate, safe_eval

# (表达式, 期望结果字符串)
GOOD_CASES = [
    ("(100 + 200) * 3", "900"),
    ("10 / 4", "2.5"),
    ("50 * 3", "150"),
    ("-5 + 3", "-2"),
    ("2 ** 10", "1024"),
    ("7 % 3", "1"),
    ("10 // 3", "3"),
    ("1.5 + 2.5", "4"),
]

# 必须被拒绝的输入：幂爆炸、除零、代码注入、属性逃逸、超长、超大、布尔
DANGEROUS_CASES = [
    "10**10**10",                          # 幂塔，会耗尽内存
    "2 ** 200",                            # 超过指数上限
    "1e300 * 1e300",                       # 结果溢出上限
    "1/0",                                 # 除零
    "1%0",                                 # 取模除零
    "__import__('os').system('echo x')",   # 代码注入
    "().__class__",                        # 属性逃逸
    "open('x')",                           # 函数调用
    "True + 1",                            # 布尔伪装成 int
    "",                                    # 空串
    "   ",                                 # 纯空白
    "a" * 300,                             # 超长表达式
    "1 + ",                                # 语法错误
]


def test_good_expressions():
    """正常四则运算必须算出正确结果。"""
    for expr, expect in GOOD_CASES:
        got = safe_calculate(expr)
        assert got == f"{expr} = {expect}", f"{expr!r} 算错了：{got}，期望尾部 = {expect}"


def test_dangerous_inputs_blocked():
    """所有危险输入必须返回 ERROR 串，绝不能真的执行或算出结果。"""
    for expr in DANGEROUS_CASES:
        got = safe_calculate(expr)
        assert got.startswith("ERROR"), f"危险输入没被拦住！{expr[:40]!r} -> {got[:60]}"


def test_safe_eval_raises_typed_error():
    """底层 safe_eval 对非法输入抛 MathEvalError（而非裸 SyntaxError/RecursionError）。"""
    for expr in DANGEROUS_CASES:
        try:
            safe_eval(expr)
        except MathEvalError:
            continue
        except RecursionError:
            raise AssertionError(f"{expr[:40]!r} 触发了递归溢出，未被提前拦截")
        else:
            raise AssertionError(f"{expr[:40]!r} 没有抛错")


def test_result_is_numeric():
    """合法表达式的返回值必须是 int/float，工具层才能安全格式化。"""
    for expr, _ in GOOD_CASES:
        from safe_math import safe_eval as _eval
        assert isinstance(_eval(expr), (int, float))


if __name__ == "__main__":
    print("=== safe_math 回归测试 ===")
    print(f"\n[1] 正常算式 {len(GOOD_CASES)} 例")
    for expr, expect in GOOD_CASES:
        got = safe_calculate(expr)
        print(f"    {'PASS' if got == f'{expr} = {expect}' else 'FAIL'}  {expr!r} -> {got}")

    print(f"\n[2] 危险输入 {len(DANGEROUS_CASES)} 例（应全部 BLOCKED）")
    for expr in DANGEROUS_CASES:
        got = safe_calculate(expr)
        flag = "BLOCKED" if got.startswith("ERROR") else "LEAKED!"
        print(f"    {flag:8} {expr[:38]!r} -> {got[:55]}")

    test_good_expressions()
    test_dangerous_inputs_blocked()
    test_safe_eval_raises_typed_error()
    test_result_is_numeric()
    print(f"\n✅ 全部 {len(GOOD_CASES) + len(DANGEROUS_CASES) + 2} 项断言通过")

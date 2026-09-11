# safe_math.py
# 安全的四则运算求值器，替代 eval()。
#
# 为什么不能用 eval()：
#   即使先用字符白名单过滤，`10**10**10**10` 这类表达式仍然合法，
#   会让 Python 去构造一个天文数字，直接把 CPU 和内存打满（拒绝服务）。
#   eval 还能通过 `().__class__` 之类的方式逃逸沙箱。
#
# 本模块用 ast 解析语法树，只放行数字和 + - * / // % ** 六种运算，
# 并对 ** 的指数、以及中间结果的大小做上限约束。
import ast
import operator

# 允许的运算符（不支持 ** 以外的任何函数调用、属性访问、下标）
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# ** 的指数上限：防止 2**999999999 这类表达式耗尽资源
_MAX_EXPONENT = 100
# 中间结果的绝对值上限：防止连乘爆炸
_MAX_RESULT = 1e18
# 表达式最长字符数
_MAX_LENGTH = 200


class MathEvalError(ValueError):
    """表达式非法或超出安全范围时抛出。"""


def _check_size(value):
    """结果过大就拒绝，避免内存被撑爆。"""
    if isinstance(value, (int, float)) and abs(value) > _MAX_RESULT:
        raise MathEvalError(f"计算结果超出允许范围（|x| > {_MAX_RESULT:g}）")
    return value


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    # 数字字面量
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):  # True/False 是 int 的子类，明确拒绝
            raise MathEvalError("不支持布尔值")
        if isinstance(node.value, (int, float)):
            return node.value
        raise MathEvalError(f"不支持的常量类型：{type(node.value).__name__}")

    # 二元运算
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BINARY_OPS:
            raise MathEvalError(f"不支持的运算符：{op_type.__name__}")
        left = _check_size(_eval_node(node.left))
        right = _check_size(_eval_node(node.right))
        if op_type is ast.Pow:
            if abs(right) > _MAX_EXPONENT:
                raise MathEvalError(f"幂指数过大（上限 {_MAX_EXPONENT}）")
        if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
            raise MathEvalError("除数不能为 0")
        return _check_size(_BINARY_OPS[op_type](left, right))

    # 一元正负号
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise MathEvalError(f"不支持的一元运算符：{op_type.__name__}")
        return _check_size(_UNARY_OPS[op_type](_eval_node(node.operand)))

    raise MathEvalError(f"不支持的语法节点：{type(node).__name__}")


def safe_eval(expression: str) -> float:
    """
    计算一个纯四则运算表达式，返回数值结果。
    非法输入一律抛 MathEvalError，不会执行任何代码。

    >>> safe_eval("(100 + 200) * 3")
    900
    """
    if not isinstance(expression, str):
        raise MathEvalError("表达式必须是字符串")
    text = expression.strip()
    if not text:
        raise MathEvalError("表达式为空")
    if len(text) > _MAX_LENGTH:
        raise MathEvalError(f"表达式过长（上限 {_MAX_LENGTH} 字符）")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as e:
        raise MathEvalError(f"表达式语法错误：{e.msg}") from e

    return _eval_node(tree)


def safe_calculate(expression: str) -> str:
    """
    工具版包装：永远返回字符串，不抛异常。
    直接给 Agent 的 calculate 工具用，失败信息也能被反思器识别为 ERROR。
    """
    try:
        result = safe_eval(expression)
    except MathEvalError as e:
        return f"ERROR: 计算失败 - {e}"
    except Exception as e:  # 兜底，保证工具不会把整个 Agent 打崩
        return f"ERROR: 计算异常 - {type(e).__name__}: {e}"

    # 整数结果不显示 .0
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"

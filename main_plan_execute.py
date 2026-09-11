# main_plan_execute.py
# Plan-and-Execute 流式 API（带进度可视化）

import asyncio
import json

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import DASHSCOPE_BASE_URL, MODEL_NAME, require_api_key
from safe_math import safe_calculate
from tool_utils import parse_plan_text, run_tool

# ========== 配置区 ==========
# ★ Key 从 .env 读取，不再硬编码
BASE_URL = DASHSCOPE_BASE_URL
# ============================

app = FastAPI(title="Plan-and-Execute Agent")
client = AsyncOpenAI(api_key=require_api_key(), base_url=BASE_URL)

# ---------- 定义工具（同之前） ----------
async def get_current_weather(location: str):
    weather_db = {
        "北京": "25°C 晴朗",
        "上海": "28°C 多云",
        "深圳": "30°C 阵雨",
    }
    return weather_db.get(location, f"ERROR: {location} 天气数据未覆盖")

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

# ---------- 规划器 ----------
async def plan(user_query: str) -> list:
    # ★ 要求输出 JSON 数组，比关键字匹配稳；parse_plan_text 还有两层退化兜底
    system_prompt = """你是一个任务规划专家。请把用户目标拆解为3~5个具体步骤。
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
    steps = parse_plan_text(plan_text)
    return steps if steps else [plan_text or user_query]

# ---------- 执行器 ----------
async def execute_step(step_desc: str, context: dict):
    context_str = "\n".join([f"之前结果：{v}" for v in context.values()])
    system_prompt = f"""已有上下文：{context_str}
根据当前步骤调用合适工具。当前步骤：{step_desc}"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"执行：{step_desc}"}
    ]
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )
    except Exception as e:
        # ★ 单步模型调用失败不该让整个 SSE 流崩掉
        return f"ERROR: 模型调用失败 - {type(e).__name__}: {e}"

    assistant_msg = response.choices[0].message
    if assistant_msg.tool_calls:
        results = []
        for tc in assistant_msg.tool_calls:
            # ★ 统一容错执行，不再因参数非法/工具缺失抛 500
            results.append(await run_tool(tc, TOOLS_MAP))
        return "\n".join(results)
    return assistant_msg.content or "执行完成（无输出）"

# ---------- 流式核心（SSE） ----------
async def plan_execute_stream(user_query: str):
    # 1. 规划阶段
    try:
        steps = await plan(user_query)
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'content': f'规划失败：{type(e).__name__}'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    yield f"data: {json.dumps({'type': 'plan', 'steps': steps}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0.1)  # 让前端感知更新

    # 2. 执行阶段
    context = {}
    for i, step in enumerate(steps, 1):
        # 发送开始执行事件
        yield f"data: {json.dumps({'type': 'step_start', 'step_index': i, 'step_desc': step}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)

        result = await execute_step(step, context)
        context[f"步骤{i}"] = result

        # 发送步骤结果
        yield f"data: {json.dumps({'type': 'step_result', 'step_index': i, 'result': result}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.1)

    # 3. 汇总阶段（发送最终答案）
    summary_prompt = f"""用户目标：{user_query}
各步骤结果：{json.dumps(context, ensure_ascii=False, indent=2)}
请生成最终回答。如果某步骤结果是 ERROR，请如实说明该步骤未完成，不要编造数据。"""
    try:
        final_resp = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.5,
        )
        final_answer = final_resp.choices[0].message.content
    except Exception as e:
        final_answer = f"⚠️ 汇总阶段调用失败（{type(e).__name__}），以下是各步骤原始结果：\n{json.dumps(context, ensure_ascii=False, indent=2)}"

    yield f"data: {json.dumps({'type': 'final', 'content': final_answer}, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"

# ---------- API 路由 ----------
class ChatRequest(BaseModel):
    user_message: str = Field(min_length=1, max_length=2000)

@app.post("/api/plan-execute/stream")
async def stream_plan_execute(req: ChatRequest):
    return StreamingResponse(
        plan_execute_stream(req.user_message),
        media_type="text/event-stream"
    )

# ---------- 可视化测试页面 ----------
@app.get("/")
async def get_test_page():
    html = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Plan-and-Execute Agent</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto; padding: 20px; background: #f5f7fa; }
            .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .plan-item { background: #e8f4fd; padding: 8px 12px; margin: 5px 0; border-radius: 5px; border-left: 4px solid #2196F3; }
            .step-log { background: #f0f0f0; padding: 8px 12px; margin: 5px 0; border-radius: 5px; font-family: monospace; }
            .step-done { background: #e8f5e9; border-left: 4px solid #4CAF50; }
            .step-running { background: #fff3e0; border-left: 4px solid #FF9800; }
            .step-failed { background: #ffebee; border-left: 4px solid #f44336; }
            #response { white-space: pre-wrap; line-height: 1.6; }
            button { background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            button:disabled { background: #90caf9; cursor: not-allowed; }
            input { width: 70%; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
            .status { color: #666; font-size: 0.9em; }
            .error { color: #c62828; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>🧠 Plan-and-Execute Agent</h2>
        <p class="status">输入复杂任务，AI 会先规划再执行，每一步都可见</p>
        <div class="card">
            <input type="text" id="msg" placeholder="例如：查一下上海天气，并计算3天打车费共多少钱" style="width:70%;">
            <button id="runBtn" onclick="sendMsg()">运行 Agent</button>
        </div>
        <div id="planArea" class="card" style="display:none;">
            <h4>📋 规划步骤</h4>
            <div id="planList"></div>
        </div>
        <div id="logArea" class="card" style="display:none;">
            <h4>🔄 执行日志</h4>
            <div id="logList"></div>
        </div>
        <div id="resultArea" class="card" style="display:none;">
            <h4>🤖 最终回答</h4>
            <div id="response"></div>
        </div>

        <script>
        async function sendMsg() {
            const msg = document.getElementById('msg').value;
            if (!msg.trim()) return;

            const btn = document.getElementById('runBtn');
            btn.disabled = true;

            // 重置界面（★ 用 textContent 清空，避免残留 HTML）
            ['planArea','logArea','resultArea'].forEach(id => document.getElementById(id).style.display = 'none');
            document.getElementById('planList').textContent = '';
            document.getElementById('logList').textContent = '';
            document.getElementById('response').textContent = '';

            let response;
            try {
                response = await fetch('/api/plan-execute/stream', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({user_message: msg})
                });
            } catch (e) {
                showError('请求失败：' + e);
                btn.disabled = false;
                return;
            }
            if (!response.ok || !response.body) {
                showError('请求失败，HTTP ' + response.status);
                btn.disabled = false;
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            try {
                while(true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, {stream: true});
                    const lines = buffer.split('\\n\\n');
                    buffer = lines.pop();

                    for (let line of lines) {
                        if (!line.startsWith('data: ')) continue;
                        const data = line.substring(6);
                        if (data === '[DONE]') continue;
                        try { handleEvent(JSON.parse(data)); } catch(e) {}
                    }
                }
            } catch (e) {
                showError('流式读取中断：' + e);
            } finally {
                btn.disabled = false;
            }
        }

        function showError(text) {
            const area = document.getElementById('resultArea');
            area.style.display = 'block';
            const el = document.getElementById('response');
            el.textContent = '';
            const span = document.createElement('span');
            span.className = 'error';
            span.textContent = text;   // ★ textContent，不解析为 HTML
            el.appendChild(span);
        }

        function handleEvent(json) {
            if (json.type === 'plan') {
                // 显示规划
                document.getElementById('planArea').style.display = 'block';
                const list = document.getElementById('planList');
                json.steps.forEach((step, idx) => {
                    const div = document.createElement('div');
                    div.className = 'plan-item';
                    // ★ textContent 而非 innerHTML：步骤文本来自模型，不可当 HTML 解析
                    div.textContent = `步骤${idx+1}: ${step}`;
                    div.id = `plan_${idx+1}`;
                    list.appendChild(div);
                });
            } else if (json.type === 'step_start') {
                // 高亮当前执行的步骤
                document.getElementById('logArea').style.display = 'block';
                const el = document.getElementById(`plan_${json.step_index}`);
                if (el) el.style.background = '#fff3e0';
                // 加日志
                const log = document.getElementById('logList');
                const div = document.createElement('div');
                div.className = 'step-log step-running';
                div.id = `log_${json.step_index}`;
                div.textContent = `⏳ 执行步骤 ${json.step_index}: ${json.step_desc}`;
                log.appendChild(div);
            } else if (json.type === 'step_result') {
                // ★ 按 step_index 精确定位日志行（原来用"最后一行"，并发/乱序时会写错行）
                const last = document.getElementById(`log_${json.step_index}`);
                const failed = typeof json.result === 'string' && json.result.includes('ERROR');
                if (last) {
                    last.className = 'step-log ' + (failed ? 'step-failed' : 'step-done');
                    last.textContent = (failed ? '❌ ' : '✅ ') + `步骤 ${json.step_index} 结果: ${json.result}`;
                }
                const el = document.getElementById(`plan_${json.step_index}`);
                if (el) el.style.background = failed ? '#ffebee' : '#e8f5e9';
            } else if (json.type === 'final') {
                document.getElementById('resultArea').style.display = 'block';
                document.getElementById('response').textContent = json.content;   // ★ textContent
            } else if (json.type === 'error') {
                showError(json.content);
            }
        }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

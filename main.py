# main.py (RAG 完整版：密钥外置 + 混合检索 + 精排 + 来源回传 + 热更新 + 前端转义)
import asyncio
import json

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL,
    ENABLE_BM25,
    ENABLE_RERANK,
    MODEL_NAME,
    RECALL_TOP_K,
    RERANK_API_MODEL,
    RERANK_BACKEND,
    VERBOSE_LOG,
    require_api_key,
    rerank_score_threshold,
)
from retriever import HybridRetriever
from ingest import ingest_directory

# ---------- 初始化 FastAPI ----------
app = FastAPI(title="RAG Agent 接口")

# ★ chat 与 embedding 共用同一个 Key（原来两处写死且不一致，embedding 会 401）
API_KEY = require_api_key()

# ---------- 初始化大模型客户端（只保留一个） ----------
client = AsyncOpenAI(api_key=API_KEY, base_url=DASHSCOPE_BASE_URL)

# ---------- 初始化 Chroma 向量库 ----------
# 使用 OpenAIEmbeddingFunction 兼容阿里云 embedding
embedding_fn = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=DASHSCOPE_BASE_URL,
)
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

# ---------- 初始化混合检索器 ----------
# 向量检索 + BM25 关键词检索 -> RRF 融合 -> 精排（API 或本地）-> 阈值过滤
# 精排后端懒加载：首次检索时才初始化，导入本模块不会拖慢启动。
retriever = HybridRetriever(collection)
_corpus_size = retriever.reload()
_backend_desc = f"API({RERANK_API_MODEL})" if RERANK_BACKEND == "api" else "本地模型"
print(f"[main] 检索器就绪：{_corpus_size} 块语料 | "
      f"BM25={'开' if ENABLE_BM25 else '关'} | 精排={'开' if ENABLE_RERANK else '关'}（{_backend_desc}）| "
      f"粗召回={RECALL_TOP_K} | 精排阈值={rerank_score_threshold()}")

# ---------- 请求模型 ----------
class ChatRequest(BaseModel):
    user_message: str
    history: list = Field(default_factory=list)   # 历史对话
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)   # 检索返回的文档块数量


# ---------- 检索函数 ----------
async def retrieve_docs(query: str, history: list, top_k: int = DEFAULT_TOP_K):
    """
    混合检索，返回 [RetrievalHit, ...]。

    ★ 相比「纯向量检索」版的三处升级：
      1) 加了一路 BM25 关键词检索，补上向量对精确标识符的盲区
         （产品型号 A100、错误码 ERR-4032、纯数字 9.9 万这类，
          语义向量几乎没有区分度，但字面匹配一抓就中）
      2) 粗召回放宽到 RECALL_TOP_K(20)，再用 CrossEncoder 精排收紧到 top_k
         ——比单纯调 top_k 有效得多，因为召回阶段宁可多捞也不漏
      3) 每条结果都带 vector_rank / bm25_rank / rrf_score / rerank_score，
         前端能看出它是被哪一路捞回来的、精排给了多少分

    仍然保留的两道闸门（各管各的信号，不互相误杀）：
      - 向量距离阈值只作用于向量召回路
      - 精排分数阈值只在精排真正生效时作为最终闸门
    """
    # ★ 将历史对话拼接到查询中，解决“那它呢”这类指代问题
    if history:
        recent_history = history[-4:]  # 取最近 4 条（2问2答）
        context_str = " ".join(
            [str(item.get("content", "")) for item in recent_history if isinstance(item, dict)]
        )
        enhanced_query = f"{context_str} {query}".strip()
    else:
        enhanced_query = query

    # ★ 集合为空时 Chroma 的 query 会直接报错，先做一次守卫
    try:
        total = collection.count()
    except Exception as e:
        print(f"读取集合失败: {e}")
        return []
    if total == 0:
        print("知识库为空，请先运行 build_index.py 建立索引。")
        return []

    # ★ 语料规模变了就重建 BM25 索引（比如中途重新跑了 build_index.py）
    if retriever._bm25 is None or retriever._bm25.doc_count != total:
        retriever.reload()

    try:
        # search() 内部把 CPU 密集的 CrossEncoder 打分丢到线程池，
        # 不会阻塞 FastAPI 的事件循环
        return await retriever.search(enhanced_query, top_k=top_k)
    except Exception as e:
        print(f"检索失败: {type(e).__name__}: {e}")
        return []


# ---------- 流式生成器（含 RAG） ----------
async def generate_stream(user_msg: str, history: list, top_k: int):
    # 1. 检索相关文档（混合检索：BM25 + 向量 + RRF 融合 + 精排）
    retrieved = await retrieve_docs(user_msg, history, top_k)
    if VERBOSE_LOG:
        for i, h in enumerate(retrieved, 1):
            print(f"  [{i}] method={h.method} rerank={h.rerank_score} "
                  f"rrf={h.rrf_score:.5f} dist={h.vector_distance} bm25={h.bm25_score}")
            print(f"      {h.text[:60]}...")

    # ★ 先把来源推给前端，做到答案可溯源
    #   额外回传各路分数与召回方式，方便直观看出混合检索与精排起了什么作用
    if retrieved:
        sources = [h.to_source_dict() for h in retrieved]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources}, ensure_ascii=False)}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'sources', 'sources': [], 'note': '知识库中未检索到相关内容（可能被相关性阈值过滤），以下回答基于模型通用知识'}, ensure_ascii=False)}\n\n"

    # 2. 构建系统 prompt
    if retrieved:
        context = "\n\n".join(
            [f"[{i}] （来源：{h.source}）\n{h.text}" for i, h in enumerate(retrieved, 1)]
        )
        system_prompt = f"""你是一个基于知识库的问答助手。请根据以下参考资料回答用户问题，如果参考资料中没有相关信息，则诚实地说“根据现有知识，我无法回答该问题”。
回答时可以用 [1] [2] 这样的编号标注信息来自哪一条参考资料。
=== 参考资料 ===
{context}
=== 参考结束 ===
"""
    else:
        system_prompt = "你是一个乐于助人的AI助手。当前知识库中没有检索到相关内容，请基于你的通用知识回答，并明确告知用户该回答未经知识库验证。"

    # 3. 组装 messages
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend([m for m in history if isinstance(m, dict) and "role" in m and "content" in m])
    messages.append({"role": "user", "content": user_msg})

    # ★ 完整 messages 只在 VERBOSE_LOG=1 时打印（含用户输入，默认不打进日志）
    if VERBOSE_LOG:
        print("\n=== 发送给大模型的完整 messages ===")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        print("====================================\n")

    # 4. 调用大模型（流式）
    try:
        stream = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=0.3   # 低温度提高事实性
        )
    except Exception as e:
        print(f"大模型调用失败: {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'模型调用失败：{type(e).__name__}'}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    # 5. 按 SSE 输出
    # ★ 加 try：流中途断开时给出明确错误事件，而不是让前端一直转圈
    try:
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"
    except Exception as e:
        print(f"流式输出中断: {type(e).__name__}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': f'输出中断：{type(e).__name__}'}, ensure_ascii=False)}\n\n"

    yield "data: [DONE]\n\n"


# ---------- 路由 ----------
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        generate_stream(req.user_message, req.history, req.top_k),
        media_type="text/event-stream",
    )


# ---------- 热更新知识库（方向 2：目录化 + 进程内重建） ----------
# 往 KNOWLEDGE_DIR 丢文件/删文件后，调一次这个接口即可生效，无需重启服务。
# ★ 为什么必须放在服务进程内：Chroma PersistentClient 官方只支持单进程使用，
#   服务跑着时另开进程跑 build_index.py 会撞 SQLite 锁或读到旧快照。
#   入库放在本进程里做，锁问题从根上不存在。
_reindex_lock = asyncio.Lock()


@app.post("/api/reindex")
async def reindex():
    if _reindex_lock.locked():
        return JSONResponse(
            status_code=409,
            content={"ok": False, "reason": "已有重建任务在进行中，请稍后再试"},
        )
    async with _reindex_lock:
        try:
            # 入库会调 embedding 接口（同步、可能几十秒），丢线程池，不卡事件循环
            result = await asyncio.to_thread(ingest_directory, collection)
            corpus = await asyncio.to_thread(retriever.reload)  # BM25 跟新库重建
        except Exception as e:
            print(f"reindex 失败: {type(e).__name__}: {e}")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "reason": f"{type(e).__name__}: {e}"},
            )
    docs = [
        {"source": d.source, "chunks": d.chunks, "action": d.action,
         **({"error": d.error} if d.error else {})}
        for d in result["documents"]
    ]
    changed = sum(1 for d in docs if d["action"] == "upserted")
    skipped = sum(1 for d in docs if d["action"] == "skipped")
    return {
        "ok": True,
        "total_chunks": result["total"],
        "corpus_size": corpus,          # BM25 已同步重建到此规模
        "changed": changed,
        "unchanged": skipped,
        "removed": result["removed_sources"],
        "documents": docs,
    }


# ---------- 测试页面（增加 top_k 与来源展示） ----------
@app.get("/")
async def get_test_page():
    html_content = """
    <html>
        <head><meta charset="utf-8"><title>RAG 智能问答测试</title></head>
        <body>
            <h2>RAG 智能问答测试</h2>
            <input type="text" id="msg" placeholder="输入问题..." style="width:300px;">
            <label style="font-size:0.85em;">top_k
              <input type="number" id="topk" value="3" min="1" max="20" style="width:50px;">
            </label>
            <button onclick="sendMsg()">发送</button>
            <p style="font-size:0.9em;color:#666;">（知识库：示例产品手册 ｜ 混合检索：BM25 + 向量 → RRF 融合 → 精排）</p>
            <h3>引用来源：</h3>
            <div id="sources" style="font-size:0.85em;color:#444;border:1px dashed #bbb;padding:8px;min-height:20px;"></div>
            <h3>AI 回复：</h3>
            <div id="response" style="white-space: pre-wrap; border: 1px solid #ccc; padding: 10px; min-height: 100px;"></div>
            <script>
                async function sendMsg() {
                    const msg = document.getElementById('msg').value;
                    if (!msg.trim()) return;
                    const resDiv = document.getElementById('response');
                    const srcDiv = document.getElementById('sources');
                    const topk = parseInt(document.getElementById('topk').value || '3', 10);
                    // ★ 用 textContent 清空，不用 innerHTML
                    resDiv.textContent = '';
                    srcDiv.textContent = '检索中...（若使用本地精排模型，首次请求需加载约 1.1GB，请稍候）';

                    let response;
                    try {
                        response = await fetch('/api/chat/stream', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({user_message: msg, top_k: topk})
                        });
                    } catch (e) {
                        srcDiv.textContent = '请求失败：' + e;
                        return;
                    }
                    if (!response.ok || !response.body) {
                        srcDiv.textContent = '请求失败，HTTP ' + response.status;
                        return;
                    }

                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';

                    while(true) {
                        const {done, value} = await reader.read();
                        if(done) break;
                        // ★ 用缓冲区处理跨 chunk 被切断的 SSE 帧（原来直接 split 会丢字）
                        buffer += decoder.decode(value, {stream: true});
                        const parts = buffer.split('\\n\\n');
                        buffer = parts.pop();

                        for(let line of parts) {
                            if(!line.startsWith('data: ')) continue;
                            const data = line.substring(6);
                            if(data === '[DONE]') continue;
                            try {
                                const evt = JSON.parse(data);
                                handleEvent(evt);
                            } catch(e) {}
                        }
                    }
                }

                function handleEvent(evt) {
                    const resDiv = document.getElementById('response');
                    const srcDiv = document.getElementById('sources');
                    if (evt.type === 'sources') {
                        // ★ 用 DOM API + textContent 渲染，模型输出与来源片段都不会被当作 HTML 执行
                        srcDiv.textContent = '';
                        if (!evt.sources || evt.sources.length === 0) {
                            srcDiv.textContent = evt.note || '（无匹配来源）';
                            return;
                        }
                        evt.sources.forEach((s, i) => {
                            const div = document.createElement('div');
                            div.style.margin = '3px 0';
                            // ★ 展示各路分数：精排分、向量距离、BM25 分、召回方式
                            //   这样能直观看出这条结果是被哪一路捞回来的、精排给了多少分
                            const parts = [];
                            if (s.rerank_score !== null && s.rerank_score !== undefined) parts.push('精排=' + s.rerank_score);
                            if (s.distance !== null && s.distance !== undefined) parts.push('向量距离=' + s.distance);
                            if (s.bm25_score !== null && s.bm25_score !== undefined) parts.push('BM25=' + s.bm25_score);
                            if (s.method) parts.push('召回=' + s.method);

                            const head = document.createElement('div');
                            head.textContent = '[' + (i+1) + '] ' + s.source + '   ' + parts.join('  ');
                            head.style.color = '#1565c0';
                            div.appendChild(head);

                            const body = document.createElement('div');
                            body.textContent = '    ' + s.snippet + '...';
                            body.style.color = '#555';
                            div.appendChild(body);

                            srcDiv.appendChild(div);
                        });
                    } else if (evt.type === 'content') {
                        resDiv.textContent += evt.content;
                    } else if (evt.type === 'error') {
                        resDiv.textContent += '\\n⚠️ ' + evt.content;
                    }
                }
            </script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

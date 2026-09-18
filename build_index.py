# build_index.py（多文档目录版）
# 命令行入口：把 KNOWLEDGE_DIR 整个目录同步进向量库。核心逻辑在 ingest.py。
#
#   venv\Scripts\python build_index.py          # 增量：只处理新增/变化的文档
#   REBUILD_INDEX=1 时全量重建：先删集合再整目录重扫
#
# ★ 注意：服务运行期间别跑本脚本（Chroma 只支持单进程写），
#   更新知识库请用 POST /api/reindex（见 README「热更新知识库」）。
import os

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
    require_api_key,
)
from ingest import ingest_directory

API_KEY = require_api_key()

embedding_fn = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=DASHSCOPE_BASE_URL,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
rebuild = os.environ.get("REBUILD_INDEX", "0") == "1"

if rebuild:
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ 已按 REBUILD_INDEX=1 删除旧集合 {COLLECTION_NAME!r}，将全量重建。")
    except Exception:
        pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)

if not os.path.isdir(KNOWLEDGE_DIR):
    raise SystemExit(
        f"❌ 知识库目录不存在：{KNOWLEDGE_DIR}\n"
        "请在该目录放入文档（txt/md/pdf），或在 .env 里把 KNOWLEDGE_DIR 指向正确路径。"
    )

result = ingest_directory(collection)

print(f"📂 知识库目录：{KNOWLEDGE_DIR}")
for d in result["documents"]:
    icon = {"skipped": "⏭️", "upserted": "✅", "failed": "❌"}[d.action]
    line = f"{icon} {d.source}: {d.chunks} 块（{d.action}）"
    if d.error:
        line += f" — {d.error}"
    print(line)
for src in result["removed_sources"]:
    print(f"🧹 已清理目录中不存在的文档残留：{src}")

changed = sum(1 for d in result["documents"] if d.action == "upserted")
if not result["documents"]:
    print("（目录中没有可入库的文件，支持格式见 .env 的 KNOWLEDGE_EXTS）")
else:
    print(f"✅ 同步完成：{changed} 个文档有更新，当前集合共 {result['total']} 个块。")

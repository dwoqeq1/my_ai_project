# build_index.py (智能分块版)
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    KNOWLEDGE_FILE,
    require_api_key,
)

# ========== 1. 配置 ==========
# ★ API Key 从 .env 读取，不再写死在代码里（见 config.py 与 .env.example）
API_KEY = require_api_key()
# =============================

# 2. 初始化 Embedding 函数（必须和 main.py 保持一致）
embedding_fn = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=DASHSCOPE_BASE_URL
)

# 3. 连接向量库
# ★ 原来无条件 delete_collection 再重建：跑一次就把已有索引全清掉。
#   改成「集合已存在就复用并做增量更新（按 id upsert）」，
#   确要全量重建时用环境变量 REBUILD_INDEX=1 显式声明。
import os

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
    embedding_function=embedding_fn
)

# 4. 读取你的文档
# ★ 加存在性检查，文件缺失时给出明确提示而不是 FileNotFoundError 栈
if not os.path.exists(KNOWLEDGE_FILE):
    raise SystemExit(
        f"❌ 找不到知识库文件：{KNOWLEDGE_FILE}\n"
        "请确认该文件存在，或在 .env 里把 KNOWLEDGE_FILE 指向正确路径。"
    )

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    text = f.read()

if not text.strip():
    raise SystemExit(f"❌ 知识库文件是空的：{KNOWLEDGE_FILE}")

# 5. ★★★ 核心升级：递归分块（自动适应中文）★★★
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # 每块约 200 个字符（适合中文语义）
    chunk_overlap=30,      # 重叠 30 个字符，防止关键信息被切断
    separators=["\n\n", "\n", "。", "，", " ", ""]  # 优先按段落、句子切
)
chunks = splitter.split_text(text)

print(f"原文档长度：{len(text)} 字符，被切分为 {len(chunks)} 个语义块。")

# 6. 生成 ID 和元数据
# ★ ID 带文件名哈希前缀，多文档入库时不会互相覆盖
import hashlib

file_tag = hashlib.md5(os.path.basename(KNOWLEDGE_FILE).encode("utf-8")).hexdigest()[:8]
source_name = os.path.basename(KNOWLEDGE_FILE)
ids = [f"{file_tag}_chunk_{i}" for i in range(len(chunks))]
metadatas = [{"source": source_name, "index": i} for i in range(len(chunks))]

# 7. 存入向量库
# ★ upsert 而非 add：重复执行脚本时不会因 ID 冲突报 DuplicateIDError
collection.upsert(
    documents=chunks,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ 数据入库成功！当前集合 {COLLECTION_NAME!r} 共 {collection.count()} 个块。")
print("前 3 块预览：")
for i in range(min(3, len(chunks))):
    print(f"块{i+1}: {chunks[i][:50]}...")

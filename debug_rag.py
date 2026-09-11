# debug_rag.py
# 向量库自检脚本：确认索引是否建好、检索是否正常
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    require_api_key,
)

# ★ API Key 从 .env 读取，不再硬编码
API_KEY = require_api_key()

# 1. 初始化 embedding 配置（必须和 build_index.py 完全一致）
embedding_fn = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=DASHSCOPE_BASE_URL
)

# 2. 连接到现有的向量库
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ★ 集合不存在时 get_collection 会直接抛异常，先列出已有集合给个明确提示
existing = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME not in existing:
    print(f"【检查结果】❌ 向量库中不存在集合 {COLLECTION_NAME!r}")
    print(f"【检查结果】当前已有集合：{existing or '（空）'}")
    print("【检查结果】请先运行 build_index.py 建立索引。")
    raise SystemExit(1)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

# 3. 查看库里到底有多少条数据
count = collection.count()
print(f"【检查结果】知识库中的文档块总数：{count}")

if count > 0:
    # 取前两条看看内容
    sample = collection.get(limit=2, include=["documents", "metadatas"])
    print("【检查结果】存储的样例内容：", sample['documents'])
    print("【检查结果】样例元数据：", sample['metadatas'])

    # 4. 模拟查询，验证检索逻辑是否正常
    # ★ 同时打印距离分数，方便判断阈值设得合不合适
    test_results = collection.query(
        query_texts=["价格"],
        n_results=min(2, count),
        include=["documents", "distances", "metadatas"],
    )
    print("【检查结果】查询'价格'返回的结果：", test_results['documents'])
    print("【检查结果】对应距离（越小越相似）：", test_results['distances'])
    print("【检查结果】对应来源：", test_results['metadatas'])
else:
    print("【检查结果】❌ 知识库是空的！这说明 build_index.py 没有成功把数据存进去。")

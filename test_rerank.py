# test_rerank.py
# 精排（Rerank）实验脚本：粗召回 top5 -> CrossEncoder 重排
#
# ⚠️ 首次运行会下载约 1GB 的精排模型，请确保磁盘与网络充足。
#    本脚本只是实验用，没有接入 main.py 主链路。
import sys

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    require_api_key,
)

# ★ 延迟导入：CrossEncoder 依赖 torch，导入即触发模型加载。
#   放在配置检查之后，能在缺依赖/缺 Key 时给出清晰提示，而不是抛一大段栈。
try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise SystemExit(
        "❌ 未安装 sentence-transformers，无法运行精排实验。\n"
        "如需运行请执行：pip install sentence-transformers\n"
        "（会连带安装 torch，体积较大；不跑精排实验可以不装）"
    )

# 1. 加载精排模型（BAAI/bge-reranker-base 是开源且支持中文的王者）
print("⏳ 正在加载精排模型（首次运行需下载约 1GB 文件，请稍候）...")
reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
print("✅ 精排模型加载完毕！")

# 2. 配置（★ Key 从 .env 读取，与 build_index.py 保持一致）
API_KEY = require_api_key()

embedding_fn = OpenAIEmbeddingFunction(
    api_key=API_KEY,
    model_name=EMBEDDING_MODEL,
    api_base=DASHSCOPE_BASE_URL
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

# ★ 集合不存在时给出明确提示
existing = [c.name for c in chroma_client.list_collections()]
if COLLECTION_NAME not in existing:
    raise SystemExit(
        f"❌ 向量库中不存在集合 {COLLECTION_NAME!r}（当前已有：{existing or '空'}）。\n"
        "请先运行 build_index.py 建立索引。"
    )

collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

if collection.count() == 0:
    raise SystemExit("❌ 知识库为空，请先运行 build_index.py。")

# 3. 模拟用户提问（★ 支持命令行传入，方便换不同问题测精排效果）
#    用法：python test_rerank.py "企业版有什么特殊权益？"
query = sys.argv[1] if len(sys.argv) > 1 else "企业版有什么特殊权益？"
print(f"\n🔍 查询：{query}")

# 4. 向量库先粗召回（取 top 5，但不超过库里的总数）
n_recall = min(5, collection.count())
results = collection.query(query_texts=[query], n_results=n_recall)
if not results['documents'] or not results['documents'][0]:
    print("未检索到资料")
    sys.exit(0)

candidates = results['documents'][0]
raw_distances = (results.get('distances') or [[]])[0]
print(f"\n【粗召回阶段】共召回 {len(candidates)} 个候选块：")
for i, doc in enumerate(candidates):
    dist = raw_distances[i] if i < len(raw_distances) else None
    dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
    print(f"  第{i+1}名 (向量距离 {dist_str}): {doc[:30]}...")

# 5. ★★★ 核心操作：精排（Rerank）★★★
# 构造 (query, passage) 对
pairs = [[query, doc] for doc in candidates]
# 计算相关性分数（分数越高越相关）
scores = reranker.predict(pairs)

# 6. 按分数从高到低排序
sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print("\n【精排（Rerank）后结果】分数越接近 1 代表越相关：")
for i, (doc, score) in enumerate(sorted_pairs):
    print(f"  第{i+1}名 (分数: {score:.4f}): {doc[:40]}...")

print("\n💡 说明：向量距离越小越相似，精排分数越大越相关，两者方向相反，别搞混。")

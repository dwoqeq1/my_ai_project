# retriever.py
# 混合检索层：向量检索 + BM25 关键词检索 -> RRF 融合 -> 精排 -> 阈值过滤
#
# ================= 为什么需要这一层 =================
# 原版 main.py 只有向量检索，存在一个固有盲区：
#   embedding 把文本压成语义向量，「语义相近」的东西聚在一起。
#   但产品型号（A100）、错误码（ERR-4032）、纯数字（9.9 万）、专有名词
#   这类「精确标识符」在语义空间里几乎没有区分度——
#   "A100 多少钱" 和 "显卡多少钱" 的向量可能很近，但用户要的是精确匹配。
#   BM25 是纯字面匹配，正好补上这块。
#
# 两路召回的分数无法直接相加（向量距离 0~2，BM25 分数无上界），
# 所以用 RRF 融合：只看「排名」不看「原始分」，天然解决量纲不可比。
#
# 融合后的顺序仍然不够准——BM25 只看字面、向量只看语义，
# 都不理解 query 和文档的「真实相关性」。精排（Cross-Encoder 思路）把
# query 和文档放在一起判断真实相关性，精度远高于双塔向量，但太贵不能全库跑。
# 所以经典三段式：粗召回放宽（20）-> 精排收紧（3）。
#
# ================= 精排双后端（RERANK_BACKEND）=================
#   api   = 阿里云百炼 qwen3-rerank。默认选择：本地零模型占用，
#           复用同一个 DASHSCOPE_API_KEY，按量计费。
#   local = 本地 CrossEncoder（bge-reranker-base，约 1.1GB 内存）。
#           适合大语料/高频检索/断网场景，不产生费用。
# 两端失败时行为一致：降级为 RRF 顺序，rerank_score 留 None，检索不崩。
#
# ================= 关键实测结论 =================
# 1) local：bge-reranker-base 是单输出模型，apply_softmax 必须为 False。
#    实测 True 时所有候选分数恒为 1.0000，排序退化为随机。
# 2) 精排的排序能力远比绝对分数可靠（local 端实测同义查询分数差 72 倍但
#    排序全对；api 端官方文档同样注明 relevance_score 不可跨请求比较）。
#    所以两套阈值都取得偏低，宁可多放不误杀。详见 config.py 注释与 README。
import asyncio
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import httpx

from config import (
    DEFAULT_TOP_K,
    ENABLE_BM25,
    ENABLE_RERANK,
    RAG_DISTANCE_THRESHOLD,
    RECALL_TOP_K,
    RERANK_API_MODEL,
    RERANK_API_TIMEOUT,
    RERANK_API_URL,
    RERANK_APPLY_SOFTMAX,
    RERANK_BACKEND,
    RERANK_MAX_LENGTH,
    RERANK_MODEL,
    RERANK_OFFLINE,
    RRF_BM25_WEIGHT,
    RRF_K,
    RRF_VECTOR_WEIGHT,
    VERBOSE_LOG,
    require_api_key,
    rerank_score_threshold,
)

# ---------- BM25 超参（Okapi BM25 业界标准值）----------
BM25_K1 = 1.5   # 词频饱和度：越大越看重重复出现的词
BM25_B = 0.75   # 长度归一化强度：0 完全不归一，1 完全按长度归一

# ---------- 分词用正则 ----------
# ASCII 标识符：字母/数字/连字符/下划线/点的连续串，整体保留（型号、错误码、版本号）
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]*")
# CJK 连续段：中文字符成片出现的地方
_CJK_RUN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

# 轻量停用词：BM25 的 IDF 本身会压低高频词，这里只去掉最没信息量的，
# 避免它们参与 n-gram 组合制造噪声（如「的价」「是什」）。
_CJK_STOPWORDS = set("的了是在我有和就不人都一上也很到说要去你会着看好这那什么怎么吗呢吧啊呀")
_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "and", "in", "for",
    "on", "with", "what", "how", "why", "who", "when", "where", "do", "does",
    "did", "be", "it", "this", "that", "i", "you", "me", "my", "your",
}


@dataclass
class RetrievalHit:
    """一条检索结果，带全部中间分数，便于观察各环节效果。"""
    text: str
    doc_id: str
    source: str = "unknown"
    chunk_index: Optional[int] = None

    # 向量检索路
    vector_rank: Optional[int] = None
    vector_distance: Optional[float] = None
    # BM25 路
    bm25_rank: Optional[int] = None
    bm25_score: Optional[float] = None
    # 融合与精排
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None

    @property
    def final_score(self) -> float:
        """最终排序依据：有精排分数用精排，否则用 RRF。"""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score

    @property
    def method(self) -> str:
        """说明这条结果是被哪几路召回的，便于调试。"""
        parts = []
        if self.vector_rank is not None:
            parts.append("vector")
        if self.bm25_rank is not None:
            parts.append("bm25")
        return "+".join(parts) if parts else "none"

    def to_source_dict(self) -> dict:
        """转成前端来源展示用的精简结构。"""
        return {
            "source": self.source,
            "snippet": self.text[:80],
            "distance": round(self.vector_distance, 4) if self.vector_distance is not None else None,
            "rerank_score": round(self.rerank_score, 4) if self.rerank_score is not None else None,
            "bm25_score": round(self.bm25_score, 4) if self.bm25_score is not None else None,
            "method": self.method,
        }


# ==========================================
# 1. 分词器：ASCII 整词 + 中文字符 n-gram
# ==========================================
def tokenize(text: str) -> list:
    """
    中英混排分词。不依赖 jieba（未安装），改用两条互补策略：

    1) ASCII 串整体保留为词：`ERR-4032` -> ['err-4032', 'err', '4032']
       既支持精确匹配，也支持拆开后的部分匹配。这是抓住型号/错误码的关键。
    2) 中文用 unigram + bigram：`企业版` -> ['企','业','版','企业','业版']
       bigram 提供词组级精度（「企业」不会被「事业」误命中），
       unigram 保证召回（即使 bigram 没对上，单字也能命中）。
       对中文检索来说，字符 n-gram 的鲁棒性优于词典分词——
       不需要维护词典，也不会因为分词错误而完全漏召。
    """
    if not text:
        return []
    tokens = []

    # --- ASCII 标识符 ---
    for m in _ASCII_TOKEN_RE.finditer(text):
        raw = m.group(0).lower().strip("._-")
        if not raw or raw in _EN_STOPWORDS:
            continue
        tokens.append(raw)
        # 含分隔符时额外拆开，让「4032」也能命中「ERR-4032」
        if any(sep in raw for sep in ".-_"):
            for part in re.split(r"[._\-]+", raw):
                part = part.strip()
                if len(part) >= 2 and part not in _EN_STOPWORDS:
                    tokens.append(part)

    # --- 中文 n-gram ---
    for run in _CJK_RUN_RE.findall(text):
        chars = [c for c in run]
        # unigram（跳过停用词单字）
        for c in chars:
            if c not in _CJK_STOPWORDS:
                tokens.append(c)
        # bigram（两端都是停用词的组合没有信息量，丢弃）
        for i in range(len(chars) - 1):
            a, b = chars[i], chars[i + 1]
            if a in _CJK_STOPWORDS and b in _CJK_STOPWORDS:
                continue
            tokens.append(a + b)

    return tokens


# ==========================================
# 2. BM25 索引（自研实现，零新增依赖）
# ==========================================
class BM25Index:
    """
    Okapi BM25 实现。不用 rank_bm25 库（未安装），公式本身很短，
    自研的好处是可以直接配合上面的中文 n-gram 分词器。

    score(q, d) = Σ_t IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
    """

    def __init__(self, documents: list):
        self.doc_count = len(documents)
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.doc_tf = [Counter(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / self.doc_count) if self.doc_count else 0.0

        # 文档频率：包含某词的文档数
        df = Counter()
        for tf in self.doc_tf:
            for term in tf:
                df[term] += 1
        # IDF 预先算好，查询时直接查表
        self.idf = {}
        for term, n in df.items():
            self.idf[term] = math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1.0)

    def search(self, query: str, top_n: int) -> list:
        """返回 [(doc_index, score), ...]，按分数降序，只含分数 > 0 的。"""
        if not self.doc_count:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        # query 里重复的词只算一次，避免同一个词反复累加
        unique_terms = set(query_tokens)

        scores = [0.0] * self.doc_count
        for term in unique_terms:
            idf = self.idf.get(term)
            if idf is None or idf <= 0:
                continue  # 语料中不存在，或出现在几乎所有文档里（无区分度）
            for i, tf in enumerate(self.doc_tf):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                # 长度归一化的分母
                norm = BM25_K1 * (1 - BM25_B + BM25_B * (self.doc_len[i] / self.avgdl if self.avgdl else 1.0))
                scores[i] += idf * (freq * (BM25_K1 + 1)) / (freq + norm)

        ranked = [(i, s) for i, s in enumerate(scores) if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:top_n]


# ==========================================
# 3. RRF 融合
# ==========================================
def rrf_fuse(rank_lists: list, weights: list) -> dict:
    """
    Reciprocal Rank Fusion。
    rank_lists: [[doc_key, ...], [doc_key, ...]]  每路一个排名列表
    weights:    [w1, w2]                           每路权重

    只用排名不用原始分数，所以「向量距离」和「BM25 分数」量纲不同也能融合。
    同一个文档被多路命中时分数相加——这正是我们想要的：两路都认为相关的，排更前。
    """
    fused = {}
    for ranks, weight in zip(rank_lists, weights):
        if weight <= 0:
            continue
        for rank, key in enumerate(ranks, start=1):  # rank 从 1 开始
            fused[key] = fused.get(key, 0.0) + weight / (RRF_K + rank)
    return fused


# ==========================================
# 4. 检索器主体
# ==========================================
class HybridRetriever:
    """
    混合检索器。持有 Chroma 集合、BM25 索引、精排模型（懒加载）。

    生命周期：进程内单例，首次调用时构建 BM25 索引；
    知识库更新后调用 reload() 重建。
    """

    def __init__(self, collection):
        self.collection = collection
        self._bm25: Optional[BM25Index] = None
        self._docs: list = []
        self._ids: list = []
        self._metas: list = []
        self._reranker = None
        self._reranker_failed = False
        # 模型加载要几秒且只能做一次，用锁避免并发请求重复加载
        self._lock = threading.Lock()

    # ---------- 语料装载 ----------
    def reload(self) -> int:
        """从 Chroma 拉取全部文档构建 BM25 索引，返回文档数。"""
        total = self.collection.count()
        if total == 0:
            self._docs, self._ids, self._metas, self._bm25 = [], [], [], None
            return 0
        data = self.collection.get(include=["documents", "metadatas"])
        self._docs = data.get("documents") or []
        self._ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        self._metas = [m if isinstance(m, dict) else {} for m in metas]
        self._bm25 = BM25Index(self._docs)
        if VERBOSE_LOG:
            print(f"[retriever] BM25 索引已构建：{len(self._docs)} 块，"
                  f"词表 {len(self._bm25.idf)} 项，平均长度 {self._bm25.avgdl:.1f}")
        return len(self._docs)

    def _ensure_corpus(self):
        if self._bm25 is None:
            self.reload()

    def _meta_of(self, i: int) -> dict:
        return self._metas[i] if i < len(self._metas) else {}

    # ---------- 精排后端懒加载 ----------
    def _get_reranker(self):
        """
        按 RERANK_BACKEND 懒加载精排后端。
        api   -> 复用一个 httpx.Client 单例（连接池常驻，省 TLS 握手）
        local -> 加载 CrossEncoder（约 1.1GB 内存）
        任何失败都置 _reranker_failed，之后长期降级为纯 RRF 排序，检索不崩。
        """
        if not ENABLE_RERANK or self._reranker is not None or self._reranker_failed:
            return self._reranker
        with self._lock:
            if self._reranker is not None or self._reranker_failed:
                return self._reranker
            try:
                if RERANK_BACKEND == "api":
                    self._reranker = httpx.Client(
                        timeout=httpx.Timeout(RERANK_API_TIMEOUT),
                        headers={
                            "Authorization": f"Bearer {require_api_key()}",
                            "Content-Type": "application/json",
                        },
                    )
                    print(f"[retriever] 精排后端=API（{RERANK_API_MODEL}），本地零模型占用")
                else:
                    if RERANK_OFFLINE:
                        # 模型已缓存，离线模式跳过联网校验，加载更快也更稳定
                        import os
                        os.environ.setdefault("HF_HUB_OFFLINE", "1")
                        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    from sentence_transformers import CrossEncoder
                    print(f"[retriever] 正在加载精排模型 {RERANK_MODEL} ...")
                    self._reranker = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
                    print(f"[retriever] 精排后端=本地模型（apply_softmax={RERANK_APPLY_SOFTMAX}）")
            except Exception as e:
                self._reranker_failed = True
                print(f"[retriever] ⚠️ 精排后端初始化失败，降级为 RRF 排序：{type(e).__name__}: {e}")
        return self._reranker

    # ---------- 各路召回 ----------
    def _vector_recall(self, query: str, top_n: int) -> list:
        """向量召回，返回 [(doc_id, distance), ...]。已在召回阶段做距离粗筛。"""
        total = self.collection.count()
        if total == 0:
            return []
        try:
            res = self.collection.query(
                query_texts=[query],
                n_results=min(top_n, total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print(f"[retriever] 向量检索失败: {type(e).__name__}: {e}")
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        ids = res.get("ids") or [[]]
        ids = ids[0] if ids and isinstance(ids[0], list) else ids

        hits = []
        for i, doc in enumerate(docs):
            if not doc or not doc.strip():
                continue
            distance = dists[i] if i < len(dists) else None
            # 距离粗筛：只作用于向量路。
            # ★ 不能拿它去过滤 BM25 的结果——BM25 命中的可能正是向量漏掉的，
            #   它的 distance 压根不存在。两道闸门各管各的信号。
            if distance is not None and distance > RAG_DISTANCE_THRESHOLD:
                if VERBOSE_LOG:
                    print(f"[retriever]   向量粗筛剔除（距离 {distance:.3f} > {RAG_DISTANCE_THRESHOLD}）：{doc[:30]}...")
                continue
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            hits.append({
                "key": ids[i] if i < len(ids) else f"idx_{i}",
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("index"),
                "distance": float(distance) if isinstance(distance, (int, float)) else None,
            })
        return hits

    def _bm25_recall(self, query: str, top_n: int) -> list:
        """BM25 召回，返回 [{key, text, source, chunk_index, score}, ...]。"""
        if not ENABLE_BM25:
            return []
        self._ensure_corpus()
        if not self._bm25 or not self._bm25.doc_count:
            return []
        try:
            ranked = self._bm25.search(query, top_n)
        except Exception as e:
            print(f"[retriever] BM25 检索失败: {type(e).__name__}: {e}")
            return []

        hits = []
        for idx, score in ranked:
            text = self._docs[idx]
            if not text or not text.strip():
                continue
            meta = self._meta_of(idx)
            hits.append({
                "key": self._ids[idx] if idx < len(self._ids) else f"idx_{idx}",
                "text": text,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("index"),
                "score": float(score),
            })
        return hits

    # ---------- 精排 ----------
    def _rerank(self, query: str, hits: list) -> list:
        """
        用 CrossEncoder 给候选打真实相关性分。
        任何异常都降级为保持 RRF 顺序（rerank_score 留空），不中断检索。
        """
        if not hits:
            return hits
        reranker = self._get_reranker()
        if reranker is None:
            return hits

        if RERANK_BACKEND == "api":
            scores = self._api_rerank_scores(reranker, query, [h["text"] for h in hits])
        else:
            pairs = [[query, h["text"]] for h in hits]
            try:
                # ★ apply_softmax 必须用 config 里的 False，理由见文件头注释
                scores = list(reranker.predict(pairs, apply_softmax=RERANK_APPLY_SOFTMAX))
            except Exception as e:
                print(f"[retriever] ⚠️ 精排打分失败，降级为 RRF 排序：{type(e).__name__}: {e}")
                return hits

        # scores 与 hits 一一对应；个别候选拿不到分数时留 None（排序垫底），不影响其余
        if scores is None:
            return hits
        for h, s in zip(hits, scores):
            if s is not None:
                h["rerank_score"] = float(s)

        before = [h["text"][:24] for h in hits]
        hits.sort(key=lambda x: x.get("rerank_score", float("-inf")), reverse=True)
        after = [h["text"][:24] for h in hits]
        if VERBOSE_LOG and before != after:
            print("[retriever]   精排调整了顺序")
        return hits

    def _api_rerank_scores(self, client, query: str, docs: list):
        """
        调用百炼 qwen3-rerank 接口，返回与 docs 等长的分数列表（按原始顺序归位）。
        失败返回 None，由调用方降级。
        官方接口：POST {RERANK_API_URL}，qwen3-rerank 用扁平请求体。
        """
        body = {
            "model": RERANK_API_MODEL,
            "query": query,
            "documents": docs,
        }
        try:
            resp = client.post(RERANK_API_URL, json=body)
            resp.raise_for_status()
            data = resp.json()
            # 兼容两种响应结构：顶层 results（qwen3-rerank）或 output.results（旧格式）
            results = data.get("results")
            if results is None:
                results = (data.get("output") or {}).get("results")
            if not isinstance(results, list):
                print(f"[retriever] ⚠️ 精排接口返回结构异常，降级为 RRF 排序：keys={list(data)}")
                return None
            scores = [None] * len(docs)
            for r in results:
                idx = r.get("index")
                if isinstance(idx, int) and 0 <= idx < len(docs):
                    scores[idx] = r.get("relevance_score")
            if VERBOSE_LOG:
                used = data.get("usage", {})
                print(f"[retriever]   精排 API 完成，候选 {len(docs)} 条，tokens={used.get('total_tokens')}")
            return scores
        except Exception as e:
            print(f"[retriever] ⚠️ 精排接口调用失败，降级为 RRF 排序：{type(e).__name__}: {e}")
            return None

    # ---------- 主入口 ----------
    def search_sync(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """
        同步检索主流程，返回 [RetrievalHit, ...]（已按最终分数排序并截断到 top_k）。

        流程：向量召回 + BM25 召回 -> RRF 融合去重 -> 精排 -> 阈值过滤 -> 截断
        """
        if not query or not query.strip():
            return []
        query = query.strip()

        total = self.collection.count()
        if total == 0:
            if VERBOSE_LOG:
                print("[retriever] 知识库为空，请先运行 build_index.py")
            return []

        recall_n = max(top_k, min(RECALL_TOP_K, total))
        vector_hits = self._vector_recall(query, recall_n)
        bm25_hits = self._bm25_recall(query, recall_n)

        if VERBOSE_LOG:
            print(f"[retriever] 召回：向量 {len(vector_hits)} 条，BM25 {len(bm25_hits)} 条")

        if not vector_hits and not bm25_hits:
            return []

        # --- 融合：以 doc_id 为 key 去重合并 ---
        pool = {}
        for h in vector_hits:
            pool[h["key"]] = {
                "key": h["key"], "text": h["text"], "source": h["source"],
                "chunk_index": h["chunk_index"],
                "distance": h.get("distance"),
            }
        for rank, h in enumerate(vector_hits, 1):
            if h["key"] in pool:
                pool[h["key"]]["vector_rank"] = rank

        for rank, h in enumerate(bm25_hits, 1):
            key = h["key"]
            if key not in pool:
                pool[key] = {
                    "key": key, "text": h["text"], "source": h["source"],
                    "chunk_index": h["chunk_index"], "distance": None,
                }
            pool[key]["bm25_rank"] = rank
            pool[key]["bm25_score"] = h.get("score")

        # --- RRF ---
        vector_order = [h["key"] for h in vector_hits]
        bm25_order = [h["key"] for h in bm25_hits]
        fused = rrf_fuse(
            [vector_order, bm25_order],
            [RRF_VECTOR_WEIGHT, RRF_BM25_WEIGHT],
        )
        for key, score in fused.items():
            if key in pool:
                pool[key]["rrf_score"] = score

        candidates = list(pool.values())
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        if VERBOSE_LOG:
            print(f"[retriever] 融合后候选 {len(candidates)} 条")

        # --- 精排 ---
        candidates = self._rerank(query, candidates)

        # --- 阈值过滤 ---
        # 只在精排真正生效时才用精排阈值当闸门；
        # 降级情况下（模型加载失败/被关闭）没有精排分数，此时用 RRF 顺序，
        # 不能拿精排阈值去卡，否则会把所有结果误杀光。
        reranked = any(c.get("rerank_score") is not None for c in candidates)
        if reranked:
            threshold = rerank_score_threshold()
            kept = []
            for c in candidates:
                score = c.get("rerank_score")
                if score is not None and score < threshold:
                    if VERBOSE_LOG:
                        print(f"[retriever]   精排阈值剔除（{score:.4f} < {threshold}）：{c['text'][:30]}...")
                    continue
                kept.append(c)
            candidates = kept

        # --- 组装输出 ---
        results = []
        for c in candidates[:top_k]:
            results.append(RetrievalHit(
                text=c["text"],
                doc_id=c["key"],
                source=c["source"],
                chunk_index=c.get("chunk_index"),
                vector_rank=c.get("vector_rank"),
                vector_distance=c.get("distance"),
                bm25_rank=c.get("bm25_rank"),
                bm25_score=c.get("bm25_score"),
                rrf_score=c.get("rrf_score", 0.0),
                rerank_score=c.get("rerank_score"),
            ))
        return results

    async def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list:
        """
        异步入口。
        ★ 精排是同步阻塞的（api 后端 = 同步 HTTP 请求；local 后端 = CPU 密集的
          模型推理），直接 await 会卡死整个事件循环（FastAPI 所有并发请求都会
          停住）。统一丢到线程池里跑。
        """
        return await asyncio.to_thread(self.search_sync, query, top_k)

    def close(self):
        """释放精排后端持有的资源（api 后端的连接池），服务退出时调用。"""
        reranker = self._reranker
        if reranker is not None and RERANK_BACKEND == "api":
            try:
                reranker.close()
            except Exception:
                pass
        self._reranker = None

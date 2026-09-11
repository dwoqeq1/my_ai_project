# test_retriever.py
# 混合检索层的回归测试。
#
# 运行方式：
#   venv\Scripts\python test_retriever.py
#   venv\Scripts\python -m pytest test_retriever.py -q
"""
这些测试刻意做到两件「不」：
  1) 不联网——用假集合桩（FakeCollection）返回预设的向量召回结果，
     不调用真实的 embedding 接口，所以跑测试不消耗 API 额度、也不受网络影响。
  2) 不加载精排模型——1.1GB 的 CrossEncoder 加载要几秒，
     单元测试里用「精排关闭 / 模型不可用」的路径来验证降级逻辑。

核心验证目标是本次改造的立论基础：
  BM25 必须能捞回向量检索漏掉的「精确标识符」（型号、错误码、纯数字）。
"""
import retriever
from retriever import (
    BM25Index,
    HybridRetriever,
    RetrievalHit,
    rrf_fuse,
    tokenize,
)

# ★ 注意：retriever.py 用 `from config import ENABLE_RERANK` 在导入时就绑定了名字，
#   改 config.ENABLE_RERANK 对它无效。测试必须直接改 retriever 模块的同名绑定。
_ORIGINAL_RERANK = retriever.ENABLE_RERANK
_ORIGINAL_BM25 = retriever.ENABLE_BM25


def _set_rerank(enabled: bool):
    retriever.ENABLE_RERANK = enabled


def _set_bm25(enabled: bool):
    retriever.ENABLE_BM25 = enabled

# ---------- 合成语料：故意混入精确标识符，制造向量检索的盲区 ----------
CORPUS = [
    "显卡 A100 售价 8.5 万元，适用于大规模训练。",
    "推理卡 T4 售价 1.2 万元，性价比很高。",
    "错误码 ERR-4032 表示显存不足，请减小 batch size。",
    "错误码 ERR-1001 表示网络连接超时。",
    "公司成立于 2018 年，总部位于杭州。",
    "我们的主产品是智能问答助手，支持私有化部署。",
    "企业版每年 9.9 万元，不限用户席位。",
    "标准版每月 999 元，包含 5 个用户席位。",
]
IDS = [f"doc_{i}" for i in range(len(CORPUS))]
METAS = [{"source": f"manual_{i}.txt", "index": i} for i in range(len(CORPUS))]


class FakeCollection:
    """
    Chroma 集合的测试桩。
    query() 返回预设的向量召回结果，用来模拟「向量检索的盲区」：
    对精确标识符查询，故意让向量路召回不到正确文档，
    这样才能证明 BM25 那一路补上了缺口。
    """

    def __init__(self, documents=None, ids=None, metas=None, vector_behavior=None):
        self._docs = documents if documents is not None else list(CORPUS)
        self._ids = ids if ids is not None else list(IDS)
        self._metas = metas if metas is not None else list(METAS)
        # vector_behavior: 可选的 query -> 命中下标列表，用于精确控制向量路结果
        self._vector_behavior = vector_behavior or {}
        self._count = len(self._docs)

    def count(self):
        return self._count

    def get(self, include=None):
        return {"documents": self._docs, "ids": self._ids, "metadatas": self._metas}

    def query(self, query_texts=None, n_results=3, include=None):
        q = (query_texts or [""])[0]
        if q in self._vector_behavior:
            idxs = self._vector_behavior[q][:n_results]
        else:
            # 默认行为：完全召不回（模拟向量对精确标识符的盲区）
            idxs = []
        return {
            "documents": [[self._docs[i] for i in idxs]],
            "ids": [[self._ids[i] for i in idxs]],
            "metadatas": [[self._metas[i] for i in idxs]],
            "distances": [[0.4 + 0.1 * k for k in range(len(idxs))]],
        }


# ==========================================
# 1. 分词器
# ==========================================
def test_tokenize_preserves_ascii_identifiers():
    """型号、错误码、版本号必须整体保留，这是 BM25 补盲区的根本前提。"""
    toks = tokenize("错误码 ERR-4032 表示显存不足")
    assert "err-4032" in toks, f"ERR-4032 没有被整体保留：{toks}"
    # 同时拆开，让「4032」也能命中
    assert "4032" in toks, f"ERR-4032 没有被拆出子串：{toks}"
    assert "err" in toks

    toks2 = tokenize("显卡 A100 售价 8.5 万元")
    assert "a100" in toks2, f"A100 丢失：{toks2}"
    assert "8.5" in toks2, f"8.5 丢失：{toks2}"


def test_tokenize_cjk_ngrams():
    """中文必须同时产出 unigram 和 bigram。"""
    toks = tokenize("企业版")
    assert "企" in toks and "业" in toks and "版" in toks, f"unigram 缺失：{toks}"
    assert "企业" in toks and "业版" in toks, f"bigram 缺失：{toks}"


def test_tokenize_drops_pure_stopwords():
    """纯停用词不应产生 token，避免噪声参与打分。"""
    assert tokenize("的了是在") == []
    assert tokenize("") == []
    assert tokenize(None) == []


def test_tokenize_case_insensitive():
    """大小写要归一，否则 'A100' 和 'a100' 匹配不上。"""
    assert tokenize("A100") == tokenize("a100")


# ==========================================
# 2. BM25 索引
# ==========================================
def test_bm25_hits_exact_identifiers():
    """BM25 必须能靠字面精确命中型号与错误码。"""
    idx = BM25Index(CORPUS)

    hits = idx.search("ERR-4032", 3)
    assert hits, "BM25 没命中 ERR-4032"
    assert hits[0][0] == 2, f"ERR-4032 应命中 doc_2，实际 doc_{hits[0][0]}"

    hits = idx.search("A100 多少钱", 3)
    assert hits and hits[0][0] == 0, f"A100 应命中 doc_0，实际 {hits[:2]}"


def test_bm25_distinguishes_similar_codes():
    """相近的错误码不能混淆：4032 与 1001 必须分开。"""
    idx = BM25Index(CORPUS)
    top = idx.search("ERR-4032", 2)
    top_ids = {i for i, _ in top}
    assert 2 in top_ids, "4032 没排在前面"
    assert 3 not in top_ids or top[0][0] == 2, "4032 与 1001 被混淆"

    top2 = idx.search("ERR-1001", 2)
    assert top2[0][0] == 3, f"1001 应命中 doc_3，实际 doc_{top2[0][0]}"


def test_bm25_ranks_rare_terms_higher():
    """IDF 应让稀有词权重更高：命中专有型号的文档排在泛泛的文档前。"""
    idx = BM25Index(CORPUS)
    hits = idx.search("T4 推理卡", 3)
    assert hits and hits[0][0] == 1, f"T4 应命中 doc_1，实际 doc_{hits[0][0]}"


def test_bm25_no_match_returns_empty():
    """语料中完全不存在的词应返回空，而不是硬凑结果。"""
    idx = BM25Index(CORPUS)
    assert idx.search("zzzqqqxxx", 3) == []
    assert idx.search("", 3) == []


def test_bm25_empty_corpus():
    """空语料不能崩。"""
    idx = BM25Index([])
    assert idx.doc_count == 0
    assert idx.search("任意查询", 3) == []


def test_bm25_scores_descending():
    """返回结果必须按分数降序。"""
    idx = BM25Index(CORPUS)
    hits = idx.search("企业版 标准版 席位 价格", 5)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True), f"分数未降序：{scores}"
    assert all(s > 0 for s in scores), "出现非正分数"


# ==========================================
# 3. RRF 融合
# ==========================================
def test_rrf_only_uses_rank_not_score():
    """RRF 只看排名不看原始分，这样量纲不同的两路才能融合。"""
    fused = rrf_fuse([["a", "b"], ["b", "a"]], [1.0, 1.0])
    # a 和 b 都是「一路第1、一路第2」，分数必然相等
    assert abs(fused["a"] - fused["b"]) < 1e-9, f"对称输入分数不等：{fused}"
    assert fused["a"] > 0


def test_rrf_multi_hit_boosts_rank():
    """被两路同时命中的文档，分数应高于只被一路命中的。"""
    fused = rrf_fuse([["both", "only_v"], ["both", "only_b"]], [1.0, 1.0])
    assert fused["both"] > fused["only_v"], "两路命中没有获得加成"
    assert fused["both"] > fused["only_b"]


def test_rrf_weights_respected():
    """权重为 0 的那一路应被完全忽略。"""
    fused = rrf_fuse([["a"], ["b"]], [1.0, 0.0])
    assert "a" in fused and "b" not in fused, f"权重 0 的路仍参与了融合：{fused}"


def test_rrf_empty_input():
    assert rrf_fuse([[], []], [1.0, 1.0]) == {}


# ==========================================
# 4. 混合检索端到端（不联网、不加载精排模型）
# ==========================================
def _make_retriever(vector_behavior=None):
    """构造一个关闭精排的检索器，纯 RRF 路径，测试可离线快速运行。"""
    _set_rerank(False)
    r = HybridRetriever(FakeCollection(vector_behavior=vector_behavior or {}))
    r.reload()
    return r


def test_hybrid_recovers_what_vector_missed():
    """
    ★ 本次改造的核心立论：向量路完全召不回时，BM25 必须把结果捞回来。
    FakeCollection 默认对所有 query 返回空的向量结果。
    """
    r = _make_retriever()
    hits = r.search_sync("ERR-4032 是什么意思", top_k=3)

    assert hits, "混合检索返回空——BM25 没有补上向量检索的盲区"
    assert all(h.vector_rank is None for h in hits), "向量路本应为空，却出现了 vector_rank"
    assert any(h.bm25_rank is not None for h in hits), "结果不是来自 BM25 路"
    assert "ERR-4032" in hits[0].text, f"首条结果不是目标文档：{hits[0].text[:50]}"
    assert hits[0].method == "bm25", f"method 标记错误：{hits[0].method}"


def test_hybrid_fuses_both_routes():
    """两路都命中同一文档时，它应排在最前，且 method 标记为 vector+bm25。"""
    # 让向量路把 doc_2（ERR-4032 那条）排第一
    r = _make_retriever(vector_behavior={"错误码 ERR-4032 显存不足": [2, 3, 0]})
    hits = r.search_sync("错误码 ERR-4032 显存不足", top_k=3)

    assert hits, "融合后无结果"
    top = hits[0]
    assert top.method == "vector+bm25", f"两路都命中却标记为 {top.method}"
    assert top.vector_rank is not None and top.bm25_rank is not None
    assert "ERR-4032" in top.text, f"两路共同命中的文档没排到第一：{top.text[:50]}"


def test_hybrid_dedupes_by_doc_id():
    """同一文档被两路召回时只能出现一次。"""
    r = _make_retriever(vector_behavior={"企业版 价格": [6, 7, 0]})
    hits = r.search_sync("企业版 价格", top_k=5)
    ids = [h.doc_id for h in hits]
    assert len(ids) == len(set(ids)), f"融合后出现重复文档：{ids}"


def test_hybrid_respects_top_k():
    r = _make_retriever()
    for k in (1, 2, 3):
        hits = r.search_sync("价格 席位 企业版 标准版", top_k=k)
        assert len(hits) <= k, f"top_k={k} 却返回 {len(hits)} 条"


def test_hybrid_empty_query_and_empty_corpus():
    """空查询、空知识库都不能崩。"""
    r = _make_retriever()
    assert r.search_sync("", top_k=3) == []
    assert r.search_sync("   ", top_k=3) == []

    empty = HybridRetriever(FakeCollection(documents=[], ids=[], metas=[]))
    empty.reload()
    assert empty.search_sync("任意问题", top_k=3) == []


def test_hybrid_scores_sorted_descending():
    r = _make_retriever()
    hits = r.search_sync("企业版 标准版 席位", top_k=5)
    scores = [h.final_score for h in hits]
    assert scores == sorted(scores, reverse=True), f"最终分数未降序：{scores}"


def test_hit_to_source_dict_shape():
    """前端来源展示所需的字段必须齐全，且分数已四舍五入。"""
    r = _make_retriever(vector_behavior={"A100 售价": [0, 1]})
    hits = r.search_sync("A100 售价", top_k=2)
    assert hits
    d = hits[0].to_source_dict()
    for key in ("source", "snippet", "distance", "rerank_score", "bm25_score", "method"):
        assert key in d, f"来源字典缺字段 {key}"
    assert isinstance(d["snippet"], str) and len(d["snippet"]) <= 80


# ==========================================
# 5. 精排降级路径（关键健壮性）
# ==========================================
def test_rerank_disabled_keeps_rrf_order():
    """精排关闭时，应完全按 RRF 分数排序，且不产生 rerank_score。"""
    r = _make_retriever()
    hits = r.search_sync("ERR-4032", top_k=3)
    assert all(h.rerank_score is None for h in hits), "精排已关闭却出现了精排分数"
    rrf = [h.rrf_score for h in hits]
    assert rrf == sorted(rrf, reverse=True)


def test_rerank_model_failure_degrades_gracefully():
    """
    精排模型加载失败时必须降级为 RRF 排序，而不是抛异常中断整个检索。
    这是线上最常见的故障场景（模型没缓存 / 显存不足 / 依赖缺失）。
    """
    _set_rerank(True)
    r = HybridRetriever(FakeCollection())
    r.reload()
    # 模拟模型加载失败：直接把失败标记打上
    r._reranker_failed = True
    r._reranker = None

    try:
        hits = r.search_sync("ERR-4032 是什么意思", top_k=3)
        assert hits, "精排不可用时检索应降级返回结果，而不是空"
        assert all(h.rerank_score is None for h in hits), "模型加载失败却仍产生精排分数"
        assert hits[0].bm25_rank is not None, "降级后应仍保留 BM25 结果"
    finally:
        _set_rerank(_ORIGINAL_RERANK)


def test_bm25_disabled_falls_back_to_vector_only():
    """关掉 BM25 后退化为纯向量检索，不应报错。"""
    _set_rerank(False)
    try:
        _set_bm25(False)
        r = HybridRetriever(FakeCollection(vector_behavior={"A100 售价": [0, 1]}))
        r.reload()
        hits = r.search_sync("A100 售价", top_k=2)
        assert hits, "纯向量模式下也应有结果"
        assert all(h.bm25_rank is None for h in hits), "BM25 已关闭却仍有 bm25_rank"
        assert all("vector" in h.method for h in hits)
    finally:
        _set_bm25(_ORIGINAL_BM25)


def test_retrieval_hit_final_score_prefers_rerank():
    """有精排分数时以精排为准，否则回落到 RRF。"""
    h = RetrievalHit(text="x", doc_id="d", rrf_score=0.01, rerank_score=0.9)
    assert h.final_score == 0.9
    h2 = RetrievalHit(text="x", doc_id="d", rrf_score=0.01, rerank_score=None)
    assert h2.final_score == 0.01


# ==========================================
# 6. 语料装载
# ==========================================
def test_reload_returns_doc_count():
    r = HybridRetriever(FakeCollection())
    assert r.reload() == len(CORPUS)
    assert r._bm25 is not None and r._bm25.doc_count == len(CORPUS)


def test_reload_handles_empty_collection():
    r = HybridRetriever(FakeCollection(documents=[], ids=[], metas=[]))
    assert r.reload() == 0
    assert r._bm25 is None


def test_metadata_fallback_when_missing():
    """元数据缺字段时应回落到默认值，而不是 KeyError。"""
    r = HybridRetriever(FakeCollection(metas=[{}] * len(CORPUS)))
    r.reload()
    hits = r.search_sync("ERR-4032", top_k=1)
    assert hits
    assert hits[0].source == "unknown", f"缺元数据时 source 应为 unknown，实际 {hits[0].source}"


# ==========================================
# 运行入口
# ==========================================
ALL_TESTS = [
    ("分词器", [
        test_tokenize_preserves_ascii_identifiers,
        test_tokenize_cjk_ngrams,
        test_tokenize_drops_pure_stopwords,
        test_tokenize_case_insensitive,
    ]),
    ("BM25 索引", [
        test_bm25_hits_exact_identifiers,
        test_bm25_distinguishes_similar_codes,
        test_bm25_ranks_rare_terms_higher,
        test_bm25_no_match_returns_empty,
        test_bm25_empty_corpus,
        test_bm25_scores_descending,
    ]),
    ("RRF 融合", [
        test_rrf_only_uses_rank_not_score,
        test_rrf_multi_hit_boosts_rank,
        test_rrf_weights_respected,
        test_rrf_empty_input,
    ]),
    ("混合检索端到端", [
        test_hybrid_recovers_what_vector_missed,
        test_hybrid_fuses_both_routes,
        test_hybrid_dedupes_by_doc_id,
        test_hybrid_respects_top_k,
        test_hybrid_empty_query_and_empty_corpus,
        test_hybrid_scores_sorted_descending,
        test_hit_to_source_dict_shape,
    ]),
    ("精排降级路径", [
        test_rerank_disabled_keeps_rrf_order,
        test_rerank_model_failure_degrades_gracefully,
        test_bm25_disabled_falls_back_to_vector_only,
        test_retrieval_hit_final_score_prefers_rerank,
    ]),
    ("语料装载", [
        test_reload_returns_doc_count,
        test_reload_handles_empty_collection,
        test_metadata_fallback_when_missing,
    ]),
]


if __name__ == "__main__":
    print("=== retriever 混合检索回归测试 ===")
    total = passed = 0
    for group, tests in ALL_TESTS:
        print(f"\n[{group}]（{len(tests)} 例）")
        for fn in tests:
            total += 1
            try:
                fn()
            except AssertionError as e:
                print(f"    FAIL  {fn.__name__}: {e}")
            except Exception as e:
                print(f"    ERROR {fn.__name__}: {type(e).__name__}: {e}")
            else:
                passed += 1
                print(f"    PASS  {fn.__name__}")

    print(f"\n汇总：{passed}/{total} 通过")
    if passed != total:
        raise SystemExit(1)
    print("✅ 全部断言通过")

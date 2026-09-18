# evaluate.py
# 检索质量评测（深化路线方向 2 的「尺子」）。
#
# 运行方式（需要 .env 配好 Key，评测会真实调用 embedding 与精排接口）：
#   venv\Scripts\python evaluate.py                    # 完整管线
#   venv\Scripts\python evaluate.py --mode no-rerank   # 关掉精排
#   venv\Scripts\python evaluate.py --mode no-bm25     # 关掉 BM25
#   venv\Scripts\python evaluate.py --mode vector-only # 退化回最初的纯向量检索
#   venv\Scripts\python evaluate.py --top-k 3          # 改评估的 k
#
# 四个模式跑一遍对比，就能用数字回答「BM25/精排到底值不值」。
# 指标定义：
#   Hit@1 / Hit@k ：期望文档出现在第 1 位 / 前 k 位（仅对有答案的用例计算）
#   MRR           ：首个命中的排名倒数取平均（衡量「排得靠不靠前」）
#   拒答正确率     ：empty_ok 用例中，检索结果为空的比例
#                   （结果非空也不算全错——若命中的恰是期望文档则单独标注）
import argparse
import asyncio
import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

import config
import retriever as retriever_mod
from config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    DASHSCOPE_BASE_URL,
    EMBEDDING_MODEL,
    require_api_key,
)
from retriever import HybridRetriever

ROOT = Path(__file__).resolve().parent


def apply_mode(mode: str) -> None:
    """
    消融开关。★ 必须改 retriever 模块里的绑定，不能只改 config：
    retriever.py 用 `from config import ENABLE_BM25 ...` 在导入时就把名字
    绑成了模块属性，改 config 上的值不会影响它（上轮 test_retriever 修过的同一课）。
    """
    if mode == "full":
        return
    if mode == "no-rerank":
        retriever_mod.ENABLE_RERANK = False
    elif mode == "no-bm25":
        retriever_mod.ENABLE_BM25 = False
    elif mode == "vector-only":
        retriever_mod.ENABLE_BM25 = False
        retriever_mod.ENABLE_RERANK = False
    else:
        raise SystemExit(f"未知模式：{mode}（可选 full/no-rerank/no-bm25/vector-only）")


async def run_eval(eval_file: Path, mode: str, top_k: int) -> int:
    spec = json.loads(eval_file.read_text(encoding="utf-8"))
    cases = spec["cases"]

    # ★ 评测必须用「无阈值」视图：管线内的阈值过滤会污染 Hit@k 的测量。
    #   做法：先照常跑（拿过滤后的结果算端到端指标），同时对未命中用例
    #   把精排阈值临时放开再跑一次，看「期望文档到底排第几、被谁挡了」。
    collection = chromadb.PersistentClient(path=CHROMA_PATH).get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=require_api_key(),
            model_name=EMBEDDING_MODEL,
            api_base=DASHSCOPE_BASE_URL,
        ),
    )
    total = collection.count()
    if total == 0:
        raise SystemExit("❌ 向量库为空，请先运行 build_index.py")
    ret = HybridRetriever(collection)
    ret.reload()

    print(f"评测模式={mode} | 用例={len(cases)} | 库={total} 块 | top_k={top_k}")
    print("=" * 100)

    hit1 = hitk = answerable = 0
    rr_sum = 0.0
    reject_ok = reject_total = 0
    miss_rows = []

    for case in cases:
        q = case["query"]
        expects = set(case.get("expect_sources") or [])
        want_empty = bool(case.get("empty_ok"))
        hits = await ret.search(q, top_k=top_k)
        sources_in_order = [h.source for h in hits]

        if want_empty:
            reject_total += 1
            ok = len(hits) == 0
            if ok:
                reject_ok += 1
            print(f"[{'拒答✔' if ok else '误答✘'}] {case['id']} {q}")
            if not ok:
                print(f"        返回了 {len(hits)} 条：{sources_in_order}")
            continue

        answerable += 1
        rank = None
        for i, s in enumerate(sources_in_order, 1):
            if s in expects:
                rank = i
                break
        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= top_k:
            hitk += 1
            rr_sum += 1.0 / rank
        flag = f"命中@{rank}" if rank else "未命中"
        print(f"[{flag:>6}] {case['id']} {q}")
        if rank is None:
            detail = "（库中无任何期望文档命中）"
            miss_rows.append((case, hits))

    # ---- 未命中用例的诊断：放开阈值重跑，看是不是被闸门误杀 ----
    if miss_rows:
        print("\n" + "-" * 100)
        print("未命中用例诊断（临时放开精排/距离阈值重跑）：")
        # ★ 精排阈值：rerank_score_threshold() 定义在 config 里，运行时读的是
        #   config 的全局名，覆盖 retriever_mod.* 无效，必须覆盖 config.*。
        #   距离阈值：retriever 直接用模块全局 RAG_DISTANCE_THRESHOLD，覆盖它即可。
        old_r = config.RERANK_SCORE_THRESHOLD_API
        old_rl = config.RERANK_SCORE_THRESHOLD_LOCAL
        old_d = retriever_mod.RAG_DISTANCE_THRESHOLD
        config.RERANK_SCORE_THRESHOLD_API = -10.0
        config.RERANK_SCORE_THRESHOLD_LOCAL = -10.0
        retriever_mod.RAG_DISTANCE_THRESHOLD = 99.0
        try:
            for case, _ in miss_rows:
                hits = await ret.search(case["query"], top_k=top_k)
                line = " | ".join(
                    f"#{i} {h.source} rr={h.rerank_score if h.rerank_score is not None else '—'}"
                    for i, h in enumerate(hits, 1)
                )
                print(f"  {case['id']} {case['query']}\n      {line}")
        finally:
            config.RERANK_SCORE_THRESHOLD_API = old_r
            config.RERANK_SCORE_THRESHOLD_LOCAL = old_rl
            retriever_mod.RAG_DISTANCE_THRESHOLD = old_d

    # ---- 汇总 ----
    print("\n" + "=" * 100)
    print(f"模式 [{mode}] 汇总")
    if answerable:
        print(f"  Hit@1        = {hit1}/{answerable} ({hit1/answerable:.0%})")
        print(f"  Hit@{top_k}       = {hitk}/{answerable} ({hitk/answerable:.0%})")
        print(f"  MRR          = {rr_sum/answerable:.3f}")
    if reject_total:
        print(f"  拒答正确率   = {reject_ok}/{reject_total} ({reject_ok/reject_total:.0%})")
    return 0 if (hitk == answerable and reject_ok == reject_total) else 2


def main():
    ap = argparse.ArgumentParser(description="RAG 检索质量评测")
    ap.add_argument("--eval-file", default=str(ROOT / "eval" / "eval_set.json"))
    ap.add_argument("--mode", default="full", choices=["full", "no-rerank", "no-bm25", "vector-only"])
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args()
    apply_mode(args.mode)
    raise SystemExit(asyncio.run(run_eval(Path(args.eval_file), args.mode, args.top_k)))


if __name__ == "__main__":
    main()

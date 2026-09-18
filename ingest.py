# ingest.py
# 多文档「增量入库」层（深化路线方向 2）。build_index.py 和 main.py 的 /api/reindex 都调这里。
#
# ================= 为什么需要这一层 =================
# 原版 build_index.py 只读一个写死的文件（KNOWLEDGE_FILE），往 knowledge/
# 里丢再多新文档也不会入库；而且没有「删掉的文档」和「改短的文档」的处理，
# 库里只增不减，陈旧块会一直参与检索、挤占 top_k 名额。
#
# 本模块的三条规则：
#   1) 稳定 ID：块 ID = md5(文档相对路径)[:8] + "_chunk_i"。
#      同一文档重跑 → ID 不变 → upsert 原地覆盖，不产生重复块；
#      子目录里的文件用相对路径算哈希，不同目录下同名文件不会互相覆盖。
#   2) 内容哈希跳变：metadata 里存整篇文档的 md5。内容和块数都没变 →
#      直接跳过，不重新调 embedding（几十页文档反复重建很烧额度）。
#   3) 垃圾清理：文档变短 → 删掉多余的尾块；文档从目录里消失 →
#      按 metadata.source 找到它所有块整体删除。库里永远只有当前目录的镜像。
#
# ★ 并发边界：Chroma 的 PersistentClient 官方定位是单进程使用。
#   所以「服务运行中更新知识库」必须走 main.py 的 /api/reindex，
#   让入库发生在服务进程内部；不要在服务跑着的时候另开进程跑 build_index.py。

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

# 文本类扩展名直接读；.pdf 走 pypdf（延迟导入，没装也不影响 txt/md 入库）
_TEXT_EXTS = {".txt", ".md", ".markdown"}


@dataclass
class DocStat:
    """单个文档的入库结果，供 CLI 和 /api/reindex 汇总展示。"""
    source: str                 # 相对知识库目录的路径（posix 风格），同时用作 metadata.source
    chunks: int = 0             # 分块数
    action: str = "skipped"     # skipped=无变化跳过 | upserted=已写入 | failed=失败
    error: str = ""             # 失败原因（action=failed 时）


# ---------- 扫描与读取 ----------

def scan_documents(root, exts=None) -> list:
    """递归扫描目录下支持的文件，返回 [(相对路径posix, Path), ...]，按路径排序保证稳定。"""
    root = Path(root)
    exts = {e.lower() for e in (exts or config.KNOWLEDGE_EXTS)}
    out = []
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append((p.relative_to(root).as_posix(), p))
    return out


def read_document(path) -> str:
    """按扩展名提取纯文本。解码顺序 utf-8-sig → utf-8 → gbk，覆盖 Windows 常见坑。"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader  # 延迟导入：只读 txt/md 的用户不需要 pypdf
        reader = PdfReader(str(path))
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")  # 单页解析失败不拖垮整篇
        return "\n".join(parts)
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------- ID 与分块 ----------

def _new_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )


def doc_fingerprint(rel_path: str) -> str:
    return hashlib.md5(rel_path.encode("utf-8")).hexdigest()[:8]


def chunk_ids_for(rel_path: str, n: int) -> list:
    """文档（相对路径, 块数）→ 稳定的块 ID 列表。"""
    tag = doc_fingerprint(rel_path)
    return [f"{tag}_chunk_{i}" for i in range(n)]


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# ---------- 入库核心 ----------

def _existing_state(collection, source: str) -> dict:
    """该文档当前在库中的 {块ID: 内容哈希}。"""
    data = collection.get(where={"source": source}, include=["metadatas"])
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    return {i: str((m or {}).get("content_hash", "")) for i, m in zip(ids, metas)}


def ingest_document(collection, rel_path: str, text: str) -> DocStat:
    """入库单个文档（纯内存入参 + 传入 collection，便于用假集合做单测）。"""
    stat = DocStat(source=rel_path)
    chunks = [c for c in _new_splitter().split_text(text) if c.strip()]
    stat.chunks = len(chunks)
    if not chunks:
        # 空文档：不该报错，而是把它的旧块全部清掉（目录是唯一真相来源）
        existing = _existing_state(collection, rel_path)
        if existing:
            collection.delete(ids=list(existing))
        stat.action = "upserted"
        return stat

    chash = content_hash(text)
    ids = chunk_ids_for(rel_path, len(chunks))
    metadatas = [
        {"source": rel_path, "chunk_index": i, "content_hash": chash}
        for i in range(len(chunks))
    ]

    existing = _existing_state(collection, rel_path)
    # 未变化：块数一致且每块的内容哈希一致 → 跳过，省一次全量 embedding
    if existing and len(existing) == len(chunks) and all(v == chash for v in existing.values()):
        stat.action = "skipped"
        return stat

    # 先写后删：中途异常最多是「新旧并存」，不会丢数据
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    stale = [i for i in existing if i not in set(ids)]  # 文档变短 → 删多余尾块
    if stale:
        collection.delete(ids=stale)
    stat.action = "upserted"
    return stat


def _delete_missing_docs(collection, current_sources: set) -> list:
    """清理目录里已不存在的文档的全部块，返回被清理的 source 列表。"""
    removed = []
    try:
        data = collection.get(include=["metadatas"])
    except Exception:
        return removed
    ids = data.get("ids") or []
    metas = data.get("metadatas") or []
    stale_ids = []
    for i, m in zip(ids, metas):
        src = str((m or {}).get("source", ""))
        if src and src not in current_sources:
            stale_ids.append(i)
            if src not in removed:
                removed.append(src)
    if stale_ids:
        collection.delete(ids=stale_ids)
    return sorted(removed)


def ingest_directory(collection, root=None, exts=None) -> dict:
    """
    把整个知识库目录同步进向量库（目录是唯一的真相来源）。
    返回 {"documents": [DocStat,...], "removed_sources": [...], "total": 块数}
    """
    root = Path(root or config.KNOWLEDGE_DIR)
    docs = scan_documents(root, exts)
    stats = []
    for rel, p in docs:
        try:
            text = read_document(p)
        except Exception as e:
            stats.append(DocStat(source=rel, action="failed",
                                 error=f"读取失败 {type(e).__name__}: {e}"))
            continue
        stats.append(ingest_document(collection, rel, text))
    removed = _delete_missing_docs(collection, {rel for rel, _ in docs})
    return {"documents": stats, "removed_sources": removed, "total": collection.count()}

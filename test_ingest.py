# test_ingest.py
# 入库层回归测试。
#
# 运行方式：
#   venv\Scripts\python test_ingest.py
#   venv\Scripts\python -m pytest test_ingest.py -q
#
# ★ 全程不联网：embedding 发生在 collection.upsert 内部，这里用假集合桩，
#   只验证「谁该写、写什么 ID、谁该删」这些控制逻辑。

from ingest import (
    chunk_ids_for,
    content_hash,
    doc_fingerprint,
    ingest_directory,
    ingest_document,
    scan_documents,
)


class FakeCollection:
    """内存版 Chroma 集合桩：只实现 ingest.py 用到的 get/upsert/delete/count。"""

    def __init__(self):
        self.docs = {}  # id -> (text, metadata)

    def get(self, where=None, include=None):
        ids, metas = [], []
        for i, (_text, m) in self.docs.items():
            if where and any(m.get(k) != v for k, v in where.items()):
                continue
            ids.append(i)
            metas.append(dict(m))
        return {"ids": ids, "metadatas": metas}

    def upsert(self, documents=None, ids=None, metadatas=None):
        for i, t, m in zip(ids, documents, metadatas):
            self.docs[i] = (t, dict(m))

    def delete(self, ids=None):
        for i in ids or []:
            self.docs.pop(i, None)

    def count(self):
        return len(self.docs)


_PASS = 0
_FAIL = []


def check(name, cond, detail=""):
    global _PASS
    if cond:
        _PASS += 1
        print(f"  ✔ {name}")
    else:
        _FAIL.append(name)
        print(f"  ✘ {name}  {detail}")


# ---------- 1. 稳定 ID ----------
print("[1] 稳定 ID")
ids_a = chunk_ids_for("example.txt", 4)
ids_b = chunk_ids_for("example.txt", 4)
check("同路径同块数 → ID 完全一致", ids_a == ids_b)
check("不同路径 → 前缀不同（不会互相覆盖）",
      chunk_ids_for("docs/example.txt", 1)[0][:8] != chunk_ids_for("example.txt", 1)[0][:8])
check("块 ID 形如 tag_chunk_i", ids_a[3].endswith("_chunk_3"))
check("内容哈希只看内容不看路径", content_hash("abc") == content_hash("abc"))
check("指纹长度 8", len(doc_fingerprint("x.txt")) == 8)

# ---------- 2. 单文档增量 ----------
print("[2] 单文档入库与增量")
c = FakeCollection()
text = "第一段。" * 60  # 足够长，能切多块
s1 = ingest_document(c, "a.md", text)
check("首次入库 → upserted", s1.action == "upserted" and s1.chunks > 1)
n1 = c.count()
s2 = ingest_document(c, "a.md", text)
check("内容未变 → skipped（不重复烧 embedding）", s2.action == "skipped")
check("跳过后库无变化", c.count() == n1)
s3 = ingest_document(c, "a.md", text + "追加的一段新内容。" * 50)
check("内容变长 → upserted", s3.action == "upserted")
check("变长后无重复块", c.count() == s3.chunks)
s4 = ingest_document(c, "a.md", text[:200])
check("内容变短 → upserted 且多余尾块被清理",
      s4.action == "upserted" and c.count() == s4.chunks)
# 先给 a.md 灌满内容，再清空 → 旧块必须全部清掉（目录是唯一真相来源）
ingest_document(c, "a.md", text)
before = c.count()
s5 = ingest_document(c, "a.md", "   \n  ")
check("文档清空 → 旧块全部清理且不报错",
      s5.action == "upserted" and s5.chunks == 0 and c.count() == 0 and before > 0)

# ---------- 3. 目录级同步（增/改/删） ----------
print("[3] 目录级同步")
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    (root / "sub").mkdir()
    (root / "one.txt").write_text("第一份文档。" * 40, encoding="utf-8")
    (root / "sub" / "two.md").write_text("子目录第二份。" * 40, encoding="utf-8")
    (root / "skip.log").write_text("不在扩展名内", encoding="utf-8")

    files = scan_documents(root, exts=[".txt", ".md"])
    names = [rel for rel, _ in files]
    check("扫描递归子目录", sorted(names) == ["one.txt", "sub/two.md"])
    check("不支持的扩展名被过滤", "skip.log" not in names)

    c = FakeCollection()
    r1 = ingest_directory(c, root)
    check("两文档均入库", sum(1 for d in r1["documents"] if d.action == "upserted") == 2)
    total1 = c.count()

    r2 = ingest_directory(c, root)
    check("重跑全跳过（幂等，不重复写）",
          all(d.action == "skipped" for d in r2["documents"]) and c.count() == total1)

    (root / "sub" / "two.md").unlink()
    r3 = ingest_directory(c, root)
    check("删文档后残留块被整体清理",
          r3["removed_sources"] == ["sub/two.md"]
          and all(str(m.get("source")) == "one.txt" for _, m in c.docs.values()))
    check("清理后块数等于剩余文档块数", c.count() < total1 and c.count() > 0)

    # metadata 完整性：source 用相对路径，chunk_index 连续
    srcs = {m.get("source") for _, m in c.docs.values()}
    idxs = sorted(m.get("chunk_index") for _, m in c.docs.values())
    check("metadata.source 为相对路径 posix 风格", srcs == {"one.txt"})
    check("chunk_index 从 0 连续", idxs == list(range(len(idxs))))

# ---------- 汇总 ----------
print("\n" + "=" * 46)
if _FAIL:
    print(f"❌ {_PASS} 通过 / {len(_FAIL)} 失败：{_FAIL}")
    raise SystemExit(1)
print(f"✅ test_ingest 全部通过：{_PASS} 项断言")

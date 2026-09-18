# config.py
# 全项目统一的配置入口：所有密钥只从这里读环境变量，禁止再写死在任何脚本里。
#
# 使用方式：
#   1. 复制 .env.example 为 .env
#   2. 在 .env 里填入你的真实 DashScope API Key
#   3. 各脚本统一 `from config import *` 或按需导入
import os
from pathlib import Path

from dotenv import load_dotenv

# 以本文件所在目录为项目根目录，保证从任何位置运行都能找到 .env
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# ---------- 大模型 ----------
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "").strip()
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
).strip()
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen-plus").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3").strip()

# ---------- 向量库与知识库 ----------
CHROMA_PATH = os.environ.get("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db")).strip()
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge").strip()

# ★ 知识库从「单文件」升级为「目录扫描」（方向 2）：
#   往 KNOWLEDGE_DIR 里丢文件（含子目录），跑 build_index.py 或 POST /api/reindex 即入库。
KNOWLEDGE_DIR = os.environ.get("KNOWLEDGE_DIR", str(PROJECT_ROOT / "knowledge")).strip()
# 支持扫描的扩展名（逗号分隔）。pdf 解析需要 pypdf，已在 requirements.txt。
KNOWLEDGE_EXTS = [
    e.strip().lower()
    for e in os.environ.get("KNOWLEDGE_EXTS", ".txt,.md,.pdf").split(",")
    if e.strip()
]

# 分块参数：块越小检索越准但上下文越碎，200/30 适合中文条款/手册类文档
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "30"))

# ---------- RAG 检索参数 ----------
# 距离阈值：Chroma 默认 L2 距离，越小越相似。超过该值的块视为不相关，不注入 prompt。
# 注意：启用精排后，精排分数才是权威的相关性闸门，本阈值只作为向量召回阶段的粗筛。
RAG_DISTANCE_THRESHOLD = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "1.0"))
DEFAULT_TOP_K = int(os.environ.get("DEFAULT_TOP_K", "3"))

# ---------- 混合检索（Hybrid Retrieval）----------
# 粗召回数量：召回阶段故意放宽（默认 20），把候选尽量捞全，
# 交给精排阶段收紧。库不足 20 块时自动按实际数量截断。
RECALL_TOP_K = int(os.environ.get("RECALL_TOP_K", "20"))

# 是否启用 BM25 关键词检索。关掉则退化为纯向量检索。
# 作用：补上向量检索对「精确标识符」的盲区（产品型号、错误码、纯数字、专有名词）。
ENABLE_BM25 = os.environ.get("ENABLE_BM25", "1").strip() not in ("0", "", "false", "False")

# RRF（Reciprocal Rank Fusion）融合常数，业界标准值 60。
# 融合公式：score = Σ weight_i / (RRF_K + rank_i)，只看排名不看原始分，
# 因此天然解决了「向量距离」和「BM25 分数」量纲不可比的问题。
RRF_K = float(os.environ.get("RRF_K", "60"))
RRF_VECTOR_WEIGHT = float(os.environ.get("RRF_VECTOR_WEIGHT", "1.0"))
RRF_BM25_WEIGHT = float(os.environ.get("RRF_BM25_WEIGHT", "1.0"))

# ---------- 精排（Rerank）----------
# 是否启用精排。关掉则只用 RRF 融合结果排序。
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "1").strip() not in ("0", "", "false", "False")

# ★ 精排后端：api（默认）| local ★
#   api   = 调用阿里云百炼 qwen3-rerank 接口。本地零模型占用（省掉约 1.1GB
#           模型与推理内存），复用同一个 DASHSCOPE_API_KEY，按量计费。
#   local = 加载本地 CrossEncoder（bge-reranker-base，约 1.1GB），
#           适合大语料/高频检索/断网场景，也不产生额外费用。
# 两种实现都保留在 retriever.py 里，改一行环境变量即可切换。
RERANK_BACKEND = os.environ.get("RERANK_BACKEND", "api").strip().lower()
if RERANK_BACKEND not in ("api", "local"):
    RERANK_BACKEND = "api"

# --- api 后端参数 ---
# 官方文档：https://help.aliyun.com/zh/model-studio/text-rerank-api
# 注意 gte-rerank（老模型）已于 2026-05-30 下线，这里用推荐的 qwen3-rerank。
RERANK_API_MODEL = os.environ.get("RERANK_API_MODEL", "qwen3-rerank").strip()
RERANK_API_URL = os.environ.get(
    "RERANK_API_URL", "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
).strip()
RERANK_API_TIMEOUT = float(os.environ.get("RERANK_API_TIMEOUT", "8"))

# --- local 后端参数（仅 RERANK_BACKEND=local 时生效）---
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base").strip()
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

# ★ 精排分数阈值（两套后端各用各的，量纲不同，不能混用）★
#
# 【local 后端】默认 0.005，由 16 条标注查询在 bge-reranker-base(logit) 上实测：
#   阈值 0.005 -> A/B 类命中 12/12，C 类正确拒答 4/4；0.05 会误杀 3 条有效查询。
#   C 类最高分 +0.0028，B 类最低有效分 +0.0137，0.005 落在中间。
#   ★★ 重要教训：精排的「排序能力」远比「绝对分数」可靠。★★
#   同一个意思的查询，绝对分数能差 72 倍，但排序全都正确。
#   所以阈值必须偏低：漏召让模型直接答不上来，误召只是多塞片段（prompt 有
#   「资料没有就说无法回答」兜底）。换语料/换模型后必须重新实测。
#
# 【api 后端】qwen3-rerank 返回 0~1 的 relevance_score。官方文档明确写着：
#   该分数是「本次请求内的相对分数，不可作为跨请求比较的绝对值」——
#   和我们在 local 上实测出的教训完全一致。
#   默认 0.4，由评测集 13 条标注查询在真实库上实测得出：
#     有效查询（A/B 类 10 条）top1 分数 >= 0.6641；
#     无关查询（C 类 3 条）  top1 分数 <= 0.2963。
#   0.4 落在空白带偏下位置（漏召代价 > 误召），两头余量约 1.35x / 1.66x。
#   ★ 上一版默认值 0.01 是拍脑袋定的，恰好低于 C 类分数带，消融评测中
#     拒答正确率从 3/3 跌到 1/3——这就是「阈值必须实测」的第二次教训。
RERANK_SCORE_THRESHOLD_LOCAL = float(os.environ.get("RERANK_SCORE_THRESHOLD_LOCAL", "0.005"))
RERANK_SCORE_THRESHOLD_API = float(os.environ.get("RERANK_SCORE_THRESHOLD_API", "0.4"))


def rerank_score_threshold() -> float:
    """按当前后端返回对应阈值（两套量纲不可混用）。"""
    return RERANK_SCORE_THRESHOLD_API if RERANK_BACKEND == "api" else RERANK_SCORE_THRESHOLD_LOCAL

# ★★ 关键：必须为 0（即使用默认 logit 模式）。★★
#   实测 bge-reranker-base 是单输出模型，apply_softmax=True 时
#   softmax 作用在单个 logit 上恒等于 1.0，所有候选分数都是 1.0000，
#   排序退化为随机（8 个测试问题全部返回同一块）。精排会静默失效。
RERANK_APPLY_SOFTMAX = os.environ.get("RERANK_APPLY_SOFTMAX", "0").strip() not in (
    "0", "", "false", "False"
)

# 模型已缓存在本地时设为 1，跳过联网校验，加载更快更稳定。
# 若本地没有该模型需联网下载，请设为 0。
RERANK_OFFLINE = os.environ.get("RERANK_OFFLINE", "1").strip() not in ("0", "", "false", "False")

# ---------- 邮件工具（day5_multi_tools.py 用） ----------
SENDER_EMAIL = os.environ.get("SENDER_EMAIL", "").strip()
SENDER_AUTH_CODE = os.environ.get("SENDER_AUTH_CODE", "").strip()
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.qq.com").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "465"))

# ---------- 调试开关 ----------
# 是否把发给大模型的完整 messages 打到控制台（生产环境建议关掉，日志里可能含用户数据）
VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "0").strip() not in ("0", "", "false", "False")


def require_api_key() -> str:
    """取 API Key；没配置就直接报清晰错误，避免脚本跑到一半才 401。"""
    if not DASHSCOPE_API_KEY:
        raise RuntimeError(
            "未检测到 DASHSCOPE_API_KEY。\n"
            f"请在 {PROJECT_ROOT / '.env'} 中填写你的 DashScope API Key"
            "（可从 .env.example 复制一份）。"
        )
    return DASHSCOPE_API_KEY


def require_sender_email() -> tuple:
    """取发件邮箱与授权码；缺配置时返回明确错误信息而不是让登录失败。"""
    if not SENDER_EMAIL or not SENDER_AUTH_CODE:
        raise RuntimeError(
            "未检测到 SENDER_EMAIL / SENDER_AUTH_CODE，无法发送邮件。\n"
            f"请在 {PROJECT_ROOT / '.env'} 中配置发件邮箱和 SMTP 授权码（注意不是登录密码）。"
        )
    return SENDER_EMAIL, SENDER_AUTH_CODE

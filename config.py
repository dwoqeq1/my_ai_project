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

# ---------- 向量库 ----------
CHROMA_PATH = os.environ.get("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db")).strip()
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "knowledge").strip()
KNOWLEDGE_FILE = os.environ.get(
    "KNOWLEDGE_FILE", str(PROJECT_ROOT / "knowledge" / "example.txt")
).strip()

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
# 是否启用 CrossEncoder 精排。关掉则只用 RRF 融合结果排序。
ENABLE_RERANK = os.environ.get("ENABLE_RERANK", "1").strip() not in ("0", "", "false", "False")
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base").strip()
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))

# ★ 精排分数阈值：默认 0.005，由 16 条标注查询实测确定（不是拍脑袋）。
#   在 knowledge/example.txt 上扫 bge-reranker-base 的分数分布：
#     阈值 0.005 -> A/B 类命中 12/12，C 类正确拒答 4/4（满分）
#     阈值 0.05  -> 只 9/12，误杀了「CTO 的名字叫什么」等 3 条有效查询
#   安全边际：C 类（应拒答）最高分 +0.0028，B 类最低有效分 +0.0137，
#            0.005 落在两者之间，两边各有约 1.8x / 2.7x 余量。
#
#   ★★ 重要教训：精排的「排序能力」远比「绝对分数」可靠。★★
#   同一个意思的查询，绝对分数能差 72 倍（「CTO 李娜」+0.9940 vs
#   「CTO 的名字叫什么」+0.0137），但排序全都把正确块排在第一。
#   所以阈值必须偏低，宁可多放一点无关片段进 prompt（system prompt 里有
#   「资料中没有就诚实说无法回答」兜底），也不要误杀有效结果——
#   漏召会让模型直接答不上来，代价比误召大得多。
#
#   换语料或换模型后必须重新实测（方法见 README「调参依据」），
#   样本要覆盖「短实体查询」这类难例，否则测不出误杀。
RERANK_SCORE_THRESHOLD = float(os.environ.get("RERANK_SCORE_THRESHOLD", "0.005"))

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

# AI Agent + RAG 学习项目

一个从零搭建 Agent 的渐进式学习项目：单次调用 → Function Calling → Plan-Execute-Reflect → RAG → FastAPI 服务化。

## 快速开始

```powershell
# 1. 安装依赖
venv\Scripts\pip install -r requirements.txt

# 2. 配置密钥：复制模板后填入你自己的 DashScope API Key
copy .env.example .env
# 然后编辑 .env，填写 DASHSCOPE_API_KEY

# 3. 建立知识库索引（RAG 前置步骤）
venv\Scripts\python build_index.py

# 4. 启动服务（二选一）
venv\Scripts\uvicorn main:app --reload              # RAG 问答，访问 http://127.0.0.1:8000
venv\Scripts\uvicorn main_plan_execute:app --reload # Plan-and-Execute，访问 http://127.0.0.1:8000
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.py` | 统一配置入口，所有密钥只从 `.env` 读取 |
| `safe_math.py` | 基于 AST 的安全四则运算，替代 `eval()` |
| `tool_utils.py` | 工具参数解析、工具执行容错、计划文本解析 |
| `retriever.py` | 混合检索层：BM25 + 向量 → RRF 融合 → CrossEncoder 精排 |
| `day1_basic.py` / `day1_stream.py` | 最小可用调用：非流式 / 流式 |
| `day5_function_calling.py` | Function Calling 单工具入门 |
| `day5_multi_tools.py` | 多工具路由（含真实发邮件，默认关闭） |
| `day6_plan_execute.py` | Plan-and-Execute 三阶段流程 |
| `day6_plan_execute_with_reflection.py` | 进阶：加反思器与失败重试 |
| `build_index.py` | 构建向量索引（分块 → embedding → Chroma） |
| `main.py` | RAG 问答服务（FastAPI + SSE + 来源回传） |
| `main_plan_execute.py` | Plan-and-Execute 服务（带步骤可视化） |
| `debug_rag.py` | 向量库自检 |
| `test_rerank.py` | CrossEncoder 精排实验（未接入主链路） |
| `test_safe_math.py` | 回归测试：安全计算（23 项断言） |
| `test_tool_utils.py` | 回归测试：计划解析 / 参数解析 / 工具容错（29 项断言） |
| `test_retriever.py` | 回归测试：分词 / BM25 / RRF 融合 / 精排降级（28 项断言，不联网） |

## 运行回归测试

不依赖 pytest，直接用项目解释器跑即可：

```powershell
venv\Scripts\python test_safe_math.py
venv\Scripts\python test_tool_utils.py
venv\Scripts\python test_retriever.py
```

装了 pytest 也可以用：`venv\Scripts\python -m pytest test_safe_math.py test_tool_utils.py test_retriever.py -q`

三组测试锁死的底线：

- `eval` 替换后，正常算式仍算对，而幂爆炸、除零、代码注入、属性逃逸、超长表达式全部被拦截；
- 工具参数为半截 JSON、工具不存在、工具内部抛异常时，都退化成可读的 `ERROR:` 串，不会让整个流程崩掉——这个格式正好能被反思器识别并触发重试；
- 规划器输出无论是 JSON 数组、`步骤1:`、`1.`、`- ` 还是带 markdown 加粗，都能解析出正确步骤数，不再整段退化成一步；
- 向量路完全召不回时，BM25 必须能把精确标识符捞回来；精排模型加载失败时检索自动降级为 RRF 排序而不是抛异常。

`test_retriever.py` 用假集合桩和合成语料，**不联网、不加载 1.1GB 精排模型**，跑起来只需几秒。

## 检索架构

`main.py` 的检索走 `retriever.py`，三段式管线：

```
用户问题
  ↓
粗召回（RECALL_TOP_K=20，故意放宽）
  ├── 向量检索：Chroma + text-embedding-v3，抓语义相近
  └── BM25 关键词检索：抓字面精确匹配
  ↓
RRF 融合去重（只看排名不看原始分，解决两路量纲不可比）
  ↓
CrossEncoder 精排（bge-reranker-base，query 与文档拼一起做交叉注意力）
  ↓
阈值过滤 → 截断到 top_k（DEFAULT_TOP_K=3）
```

**为什么需要 BM25 这一路**：embedding 把文本压成语义向量，但产品型号（`A100`）、错误码（`ERR-4032`）、纯数字（`9.9 万`）、邮箱电话这类「精确标识符」在语义空间里几乎没有区分度——「A100 多少钱」和「显卡多少钱」的向量很近，但用户要的是精确匹配。BM25 是纯字面匹配，正好补上这块。

**分词策略**（不依赖 jieba，零新增依赖）：

- ASCII 串整体保留并额外拆分：`ERR-4032` → `['err-4032', 'err', '4032']`，既支持精确匹配也支持部分匹配
- 中文用 unigram + bigram：`企业版` → `['企','业','版','企业','业版']`。bigram 提供词组精度（「企业」不会被「事业」误命中），unigram 保证召回。对中文检索来说字符 n-gram 比词典分词更鲁棒——不维护词典，也不会因分词错误完全漏召

**两个可能踩的坑**（都已在代码里处理）：

1. `CrossEncoder.predict` 的 `apply_softmax` **必须为 False**。bge-reranker-base 是单输出模型，softmax 作用在单个 logit 上恒等于 1.0，所有候选分数都变成 `1.0000`，排序退化为随机——实测 8 个问题全部返回同一块。精排会**静默失效**，不报任何错。
2. 精排是 CPU 密集的阻塞调用，直接 `await` 会卡死 FastAPI 事件循环（所有并发请求一起停住）。`retriever.search()` 用 `asyncio.to_thread` 丢到线程池。

## 调参依据

`RERANK_SCORE_THRESHOLD` 默认 `0.005`，由 16 条人工标注查询实测得出：

| 阈值 | A/B 类命中（语义+实体） | C 类正确拒答 | 综合 |
| --- | --- | --- | --- |
| 0.005 | **12/12** | **4/4** | 16 |
| 0.02 | 10/12 | 4/4 | 14 |
| 0.05 | 9/12 | 4/4 | 13 |
| 0.1 | 8/12 | 4/4 | 12 |

安全边际：C 类（知识库没有的内容）最高分 `+0.0028`，B 类（实体查询）最低有效分 `+0.0137`，`0.005` 落在两者中间。

**★ 最重要的结论：精排的「排序能力」远比「绝对分数」可靠。**

同一个意思的查询，绝对分数能差 72 倍——「CTO 李娜」得 `+0.9940`，「CTO 的名字叫什么」只得 `+0.0137`——但排序**全部**把正确块排在第一位。所以：

- 阈值必须偏低。漏召会让模型直接答不上来，误召只是多塞一点无关片段（system prompt 里有「资料中没有就诚实说无法回答」兜底）。**漏召的代价远大于误召。**
- 换语料或换模型后必须重新实测，且样本要覆盖「短实体查询」这类难例，否则测不出误杀。方法：准备标注查询集（含应拒答的负例），扫阈值看召回/拒答双指标。

⚠️ 当前结论建立在 **4 块语料**上，样本太小。语料规模上去后阈值需要重测——这也是「方向 2：真实语料 + 评测集」值得优先做的原因。

## 安全说明

- `.env` 已被 `.gitignore` 排除，**绝不要提交或外发**。
- `_原始备份_勿提交/` 目录里保存着改造前的代码，其中含历史明文密钥，仅供本地回滚，同样已在 `.gitignore` 中排除。
- `calculate` 工具使用 `safe_math.safe_calculate`，只放行四则运算，对幂指数和结果大小设上限。
- `day5_multi_tools.py` 的邮件用例会真实发信且不可撤回，默认跳过；需测试时设 `ENABLE_EMAIL_DEMO=1`。
- `build_index.py` 默认增量更新（upsert），不会清库；确需全量重建时设 `REBUILD_INDEX=1`。

## 可调环境变量

见 `.env.example`，常用的几个：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `qwen-plus` | 对话模型 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 向量化模型 |
| `RAG_DISTANCE_THRESHOLD` | `1.0` | 检索距离阈值，越小越严格；超阈值片段不注入 prompt |
| `DEFAULT_TOP_K` | `3` | 默认召回片段数 |
| `VERBOSE_LOG` | `0` | 设为 `1` 时把完整 messages 打到控制台（含用户输入，生产环境建议关） |

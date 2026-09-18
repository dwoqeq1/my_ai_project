# AI Agent + RAG 项目

一个从零开始搭 Agent 的学习项目。走过的路径：单次调用、Function Calling、Plan-Execute-Reflect、RAG，最后是 FastAPI 服务化。

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
| `retriever.py` | 混合检索层：BM25 + 向量，RRF 融合，再接精排（API/本地双后端） |
| `ingest.py` | 多文档入库层：目录扫描、稳定 ID、增量 upsert、清理已删块 |
| `day1_basic.py` / `day1_stream.py` | 最小可用调用：非流式 / 流式 |
| `day5_function_calling.py` | Function Calling 单工具入门 |
| `day5_multi_tools.py` | 多工具路由（含真实发邮件，默认关闭） |
| `day6_plan_execute.py` | Plan-and-Execute 三阶段流程 |
| `day6_plan_execute_with_reflection.py` | 进阶：加反思器与失败重试 |
| `build_index.py` | 构建向量索引：扫描 knowledge/ 目录，分块、embedding、写入 Chroma |
| `main.py` | RAG 问答服务（FastAPI + SSE + 来源回传 + /api/reindex 热更新） |
| `main_plan_execute.py` | Plan-and-Execute 服务（带步骤可视化） |
| `evaluate.py` | 检索质量评测（Hit@k / MRR / 拒答正确率，支持消融模式） |
| `eval/eval_set.json` | 评测集（13 条标注用例，含防回归用例） |
| `debug_rag.py` | 向量库自检 |
| `test_rerank.py` | CrossEncoder 精排实验（未接入主链路） |
| `test_safe_math.py` | 回归测试：安全计算（23 项断言） |
| `test_tool_utils.py` | 回归测试：计划解析 / 参数解析 / 工具容错（29 项断言） |
| `test_retriever.py` | 回归测试：分词 / BM25 / RRF 融合 / 精排降级（28 项断言，不联网） |
| `test_ingest.py` | 回归测试：稳定 ID / 增量跳过 / 变短清理 / 幂等（20 项断言，不联网） |

## 运行回归测试

不依赖 pytest，直接用项目解释器跑：

```powershell
venv\Scripts\python test_safe_math.py
venv\Scripts\python test_tool_utils.py
venv\Scripts\python test_retriever.py
venv\Scripts\python test_ingest.py
```

装了 pytest 也可以：`venv\Scripts\python -m pytest test_safe_math.py test_tool_utils.py test_retriever.py test_ingest.py -q`

四组测试分别盯住这些行为：

- `eval` 换掉后，正常算式照样算对；幂爆炸、除零、代码注入、属性逃逸、超长表达式全部拦截。
- 工具参数是半截 JSON、工具不存在、工具内部抛异常时，都退化成可读的 `ERROR:` 串，不会把流程打崩。反思器认这个格式，能据此触发重试。
- 规划器输出无论是 JSON 数组、`步骤1:`、`1.`、`- ` 还是带 markdown 加粗，都能解析出正确的步骤数，不会整段退化成一步。
- 向量路召不回时，BM25 必须能把精确标识符捞回来；精排模型加载失败时，检索自动降级成 RRF 排序，而不是抛异常。
- 入库层：同一文档重跑 ID 稳定、不产生重复块；内容没变就跳过，不烧 embedding；文档变短或清空时多余块清掉；重跑幂等。

`test_retriever.py` 和 `test_ingest.py` 用假集合桩与合成语料，不联网、不加载精排模型，几秒跑完。

## 更新知识库（热更新）

把文件放进 `knowledge/` 目录（含子目录）就行，支持 `.txt / .md / .pdf`，可用 `KNOWLEDGE_EXTS` 改。

```powershell
# 服务没在跑：命令行同步
venv\Scripts\python build_index.py

# 服务正在跑：调接口热更新，立即生效，不用重启
curl -X POST http://127.0.0.1:8000/api/reindex
```

`/api/reindex` 会返回每个文档的处理结果（`upserted / skipped / failed`）、清理掉的已删文档、当前总块数。

两条使用上的约束：

1. 服务运行期间不要另开进程跑 `build_index.py`。Chroma 的 `PersistentClient` 只支持单进程，跨进程写会撞 SQLite 锁或者读到旧快照。运行中的更新一律走 `/api/reindex`，入库发生在服务进程内部。
2. 改知识不会全量重烧 embedding。块 ID 等于文档路径哈希加序号，内容哈希没变的文档直接跳过；删掉的、改短的文档，旧块自动清理，库里的内容始终是目录的镜像。

## 检索质量评测

改检索参数之前先跑评测，指标来自实测而不是感觉。评测会真实调用 embedding 与精排接口：

```powershell
venv\Scripts\python evaluate.py                      # 完整管线
venv\Scripts\python evaluate.py --mode no-rerank     # 消融：关精排
venv\Scripts\python evaluate.py --mode no-bm25       # 消融：关 BM25
venv\Scripts\python evaluate.py --mode vector-only   # 退化回最初的纯向量检索
```

输出 Hit@1、Hit@k、MRR、拒答正确率四个指标。没命中的用例会临时放开阈值自动重跑一次，用来区分两种情况：是排序没排上来，还是被阈值误杀了。四个模式各跑一遍，BM25 和精排各自贡献了多少就能量化出来。

用例格式见 `eval/eval_set.json`。加新文档时顺手补几条用例：正例填 `expect_sources`（期望命中的相对路径），知识库里没有答案的负例填 `"empty_ok": true`。负例别删：阈值调高了会误杀、调低了会硬塞无关片段，两种毛病都靠负例暴露。

## 检索架构

`main.py` 的检索调用 `retriever.py`，管线三段：

```
用户问题
  ↓
粗召回（RECALL_TOP_K=20，故意放宽）
  ├── 向量检索：Chroma + text-embedding-v3，负责语义相近
  └── BM25 关键词检索：负责字面精确匹配
  ↓
RRF 融合去重（只看排名不看原始分，两路量纲不同也没关系）
  ↓
精排（双后端，RERANK_BACKEND 一键切换）
  ├── api   ：阿里云 qwen3-rerank（默认，本地零占用，返回 0~1 相对分）
  └── local ：本地 bge-reranker-base（约 1.1GB 内存，免费，首次加载约 7 秒）
  ↓
阈值过滤（两套后端量纲不同，阈值各自独立）→ 截断到 top_k（DEFAULT_TOP_K=3）
```

为什么要有 BM25 这一路：embedding 把文本压成向量，擅长找语义相近的内容，但产品型号（`A100`）、错误码（`ERR-4032`）、纯数字、邮箱电话这类精确标识符在语义空间里几乎没有区分度。"A100 多少钱"和"显卡多少钱"的向量很近，可用户要的是精确匹配。BM25 是字面匹配，补的正是这块。

分词策略（不依赖 jieba，零新增依赖）：

- ASCII 串整体保留并额外拆分：`ERR-4032` → `['err-4032', 'err', '4032']`，精确匹配和部分匹配都支持。
- 中文用 unigram + bigram：`企业版` → `['企','业','版','企业','业版']`。bigram 管词组精度，"企业"不会被"事业"误命中；unigram 管召回。对中文检索来说字符 n-gram 比词典分词更稳：不用维护词典，也不会因为分词错误整条漏召。

两个已经踩到、也已在代码里处理掉的坑：

1. `CrossEncoder.predict` 的 `apply_softmax` 必须为 False。bge-reranker-base 是单输出模型，softmax 作用在单个 logit 上恒等于 1.0，所有候选的分数都变成 `1.0000`，排序退化成随机，实测 8 个问题全部返回同一块。精排不报错，只是悄悄失效。
2. 精排是阻塞调用（本地后端是 CPU 密集推理），直接 `await` 会卡死 FastAPI 事件循环，所有并发请求一起停住。`retriever.search()` 统一用 `asyncio.to_thread` 丢进线程池。

## 调参依据

阈值都是实测出来的。两套后端量纲不同，数据分开记录。

local 后端（bge-reranker-base，logit 量纲），默认 0.005。16 条人工标注查询扫出来的结果：

| 阈值 | A/B 类命中（语义+实体） | C 类正确拒答 | 综合 |
| --- | --- | --- | --- |
| 0.005 | **12/12** | **4/4** | 16 |
| 0.02 | 10/12 | 4/4 | 14 |
| 0.05 | 9/12 | 4/4 | 13 |
| 0.1 | 8/12 | 4/4 | 12 |

安全边际：C 类（知识库里没有的内容）最高分 `+0.0028`，B 类（实体查询）最低有效分 `+0.0137`，0.005 在两者之间。

api 后端（qwen3-rerank，0~1 相对分），默认 0.4。评测集 13 条查询实测：有效查询（A 语义加 B 实体共 10 条）top1 分数都在 `0.6641` 以上，无关查询（C 类 3 条）top1 都在 `0.2963` 以下。0.4 落在两组之间的空白带偏下位置，两头余量约 1.35x / 1.66x。

api 阈值这里有过一次教训：接入初期默认值 0.01 没实测就定了，恰好低于无关查询的分数带。消融评测马上抓到问题：开 BM25 时拒答正确率从 3/3 掉到 1/3。BM25 捞回来的无关片段精排给到 0.24 上下，0.01 挡不住。

由这两组数据得出的共同结论：精排的排序可靠，绝对分数不可靠。同一个意思的查询，绝对分数能差 72 倍，"CTO 李娜"得 `+0.9940`，"CTO 的名字叫什么"只有 `+0.0137`，但排序两次都把正确块放在第一。所以阈值宁可偏低：漏召会让模型直接答不上来，误召只是多塞几段无关内容进 prompt，system prompt 里还有"资料中没有就诚实说无法回答"兜底。换语料或换模型后要重新实测，样本记得覆盖短实体查询这类难例，不然测不出误杀。直接跑 `venv\Scripts\python evaluate.py` 就行，评测集里的负例别删。

目前的结论仍建立在很小的语料上（example.txt 切出 3 块），BM25 在这个规模上看不出优势。语料上去之后阈值需要重测。多文档入库和热更新已经就位，往 `knowledge/` 放真实文档、补评测用例即可。

## 安全说明

- `.env` 已被 `.gitignore` 排除，不要提交，也不要外发。
- `_原始备份_勿提交/` 保存着改造前的代码，里面有历史明文密钥，只用于本地回滚，同样已被 `.gitignore` 排除。
- `calculate` 工具使用 `safe_math.safe_calculate`，只放行四则运算，幂指数和结果大小都有上限。
- `day5_multi_tools.py` 的邮件用例会真实发信且不可撤回，默认跳过；要测试时设 `ENABLE_EMAIL_DEMO=1`。
- `build_index.py` 默认增量更新（upsert），不会清库；确需全量重建时设 `REBUILD_INDEX=1`。

## 可调环境变量

完整列表见 `.env.example`，常用的几个：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MODEL_NAME` | `qwen-plus` | 对话模型 |
| `EMBEDDING_MODEL` | `text-embedding-v3` | 向量化模型 |
| `KNOWLEDGE_DIR` | `./knowledge` | 知识库目录（递归扫描 txt/md/pdf） |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `30` | 分块参数 |
| `RAG_DISTANCE_THRESHOLD` | `1.0` | 向量召回的距离阈值，越小越严格；超阈值片段不注入 prompt |
| `DEFAULT_TOP_K` | `3` | 最终返回给模型的片段数 |
| `RERANK_BACKEND` | `api` | 精排后端：`api` 走阿里云 qwen3-rerank，`local` 用本地 1.1GB 模型 |
| `RERANK_SCORE_THRESHOLD_API` | `0.4` | api 后端精排阈值（qwen3-rerank 返回 0~1 相对分，实测校准） |
| `RERANK_SCORE_THRESHOLD_LOCAL` | `0.005` | local 后端精排阈值（logit 量纲，16 条查询实测得出） |
| `VERBOSE_LOG` | `0` | 设为 `1` 时把完整 messages 打到控制台（含用户输入，生产环境建议关） |

# AgentMesh Arena — Architecture Decisions Log

> 本文件记录了项目从构思到实现过程中所有关键的设计决策、放弃的路线和背后的原因。
> 每次开启新的 Claude Code 会话时，请将此文件附在启动 prompt 中。
> 最后更新：2026-06-26

---

## 一、项目定位（最终确认版）

### 核心定位
AgentMesh 是一个**开放擂台（Arena）**，不是一个"更好的 trading agent"。

它的价值是：给定同一组历史数据，让来自 GitHub 上不同的 trading agent 框架各自给出答案，并排展示结果，让用户自己判断哪种视角适合自己的决策场景。

**一句话定位：**
> "不同 trading agent 面对同一个金融问题的不同见解——分歧本身就是信息。"

### 明确不是什么
- 不是"训练一个更强的 agent"
- 不是单一的 trading 系统
- 不是高频交易工具
- 不是声称能预测市场的产品

### 目标用户
- GitHub 上各 trading agent 项目的开发者（想知道自己的框架和别人比怎么样）
- 量化研究者（需要标准化的跨框架对比基准）
- 对 AI trading 感兴趣的开发者（想找到适合自己场景的工具）

---

## 二、核心架构决策

### 决策 1：容器化隔离，而不是统一依赖环境

**背景：** 各框架依赖严重冲突。TradingAgents 需要 Python≥3.12 + langchain 全家桶；FinRL 需要较老版本的 gym + stable-baselines3；未来接入更多框架冲突只会更严重。

**决定：** 每个 agent 框架运行在独立的 Docker 容器里，对外统一暴露一个 HTTP 接口。

**统一接口协议：**
```
POST /signal
输入：{"ticker": "AAPL", "date": "2024-01-15", "data": {...}}
输出：{"action": "BUY", "confidence": 0.82, "reasoning": "..."}
```

**放弃的路线：** 把所有框架装进同一个 conda 环境。原因：在实际安装中已验证不可行（tiktoken 需要 Rust 编译器，FinRL 和 TradingAgents 的依赖树有版本冲突）。

**接入新框架的标准流程：**
1. 写一个 `Dockerfile`（10-20行）
2. 写一个 `wrapper.py` 把框架原生调用包装成 `POST /signal`（30-50行）
3. 在 `docker-compose.yml` 里加一个 service 条目（5行）

### 决策 2：可插拔架构（BaseAgent + OptimizerRegistry）

**背景：** 需要让第三方开发者能以最低成本接入自己的 agent，不需要了解 AgentMesh 内部代码。

**决定：** 定义 `BaseAgent` 抽象接口和 `OptimizerRegistry` 注册表。

```python
class BaseAgent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def decide(self, ticker: str, date: str, data: dict) -> Signal: ...

# 接入只需3步：
@OptimizerRegistry.register          # 1. 加装饰器
class MyAgent(BaseAgent):
    @property
    def name(self): return "my_agent" # 2. 给名字
    def decide(self, ...): ...         # 3. 实现逻辑
```

**当前局限（v0.1）：** 用户仍需写 Python 类。路线图：v0.2 支持配置文件声明式接入；v0.3 支持纯 HTTP endpoint 注册。

### 决策 3：数据层独立，各框架共享同一份原始数据

**背景：** 公平对比的前提是所有框架看到相同的原始数据。

**决定：** 统一用 yfinance 下载标准化 OHLCV CSV，存入 `data/raw/`，所有框架共享这份数据。

**各框架数据处理差异：**
- FinRL：被动接收，需要预计算技术指标（MACD/RSI/布林带）转成 numpy array，维度必须和训练时一致
- TradingAgents：主动抓取，自带数据层（Alpha Vantage + yfinance + Reddit/StockTwits），需要控制日期边界防止 look-ahead bias
- 已知风险：yfinance 是逆向工程爬虫，不是官方 API，存在稳定性风险。数据层设计成 `BaseDataSource` 接口以便将来替换为 Alpha Vantage。

**关键注意：** TradingAgents 必须使用 v0.2.3 及以上版本，该版本修复了 look-ahead bias（回测日期保真度）问题。

### 决策 4：FinRL 展示策略，不展示信号

**背景：** FinRL 的原生输出是多股票组合的仓位权重向量，不是单股 BUY/HOLD/SELL 信号。

**放弃的路线：** 用 `±0.1` 阈值把 FinRL 权重强行离散化成 BUY/HOLD/SELL。

**原因：** 这导致了实验里 FinRL 整个测试期只在第一天输出一次 BUY 的问题——FinRL 每天都在小幅调仓，但变化量全部落在阈值死区里被标记成 HOLD，完全掩盖了 FinRL 的真实行为。

**决定：** FinRL 展示每日组合权重矩阵（5只股票+现金，每天一行），以及累计收益曲线、Sharpe/MDD/Calmar。"信号一致性"分析中，用 FinRL 仓位变化方向（权重增加=看多，减少=看空，变化<1%=中性）和 TradingAgents 的 BUY/HOLD/SELL 对比。

### 决策 5：两个框架用不同展示维度，不强行用同一把尺子

**决定：**

FinRL 展示（组合策略维度）：
- 5只股票动态仓位权重曲线
- 60天累计收益 vs Buy-and-Hold
- Sharpe / MDD / Calmar / 换手率
- 哪只股票被超配/低配

TradingAgents 展示（信号质量维度）：
- 每日推理链摘要（4个分析师结论句 + 最终决策理由）
- 5只股票的信号分布和置信度变化
- 对特定新闻事件的反应
- 不强求计算 Sharpe（样本量和成本都不支撑）

两者共同展示（擂台独有价值）：
- 信号一致性热力图（5只股票 × 60天）
- 分歧最大的10个交易日复盘
- 决策时间对比（毫秒 vs 7.3分钟）
- 运行成本对比（~$0 vs $0.027-$0.34/天）

---

## 三、已验证的关键数字（真实测量值）

### TradingAgents 真实成本（2026-06-18 实测）
```
单次决策（1只股票，1个交易日）：
- LLM 调用次数：18次
- Token 消耗：162,821（131K prompt / 31K completion）
- 耗时：435.9秒（7.3分钟）
- 成本：$0.027（all-flash）～ $0.338（all-pro）

扩展到 5只股票 × 60天（300个决策点）：
- 成本：$8.1（最低，全用flash）～ $101（最高，全用pro）
- 串行耗时：约36.5小时
- 建议：全程用 DeepSeek-flash，预计总成本 $10-15
```

### FinRL 成本
```
训练：GPU 约 1-2小时（RTX 8000）
推理：几乎为零（毫秒级，无 API 调用）
```

### TradingAgents 输出词汇
```
原生输出是组合经理风格评级：Underweight / Hold / Overweight
映射到标准格式：SELL / HOLD / BUY
```

---

## 四、实验设计决策

### 当前实验规模（最终确认）
- 股票池：AAPL、MSFT、NVDA、GOOGL、AMZN（5只 S&P500 成分股）
- 测试期：2024-01-01 至 2024-03-31（60个交易日）
- FinRL 训练期：2020-01-01 至 2022-12-31
- 初始资金：$100,000（虚拟）
- 交易成本：0.1%/笔

### 为什么是 60 天而不是更长
- FinRL 的 Sharpe ratio 在 60 天样本下开始有统计意义（最短窗口）
- TradingAgents 的成本在 60 天内可控（$10-15）
- 更长的测试期留给社区贡献者用自己的 API key 完成

### 放弃的统计工作（转为 good-first-issue）
- bootstrap 置信区间（需要多次重复实验）
- Jobson-Korkie 检验（Sharpe ratio 显著性检验）
- FinRL 多随机种子训练（验证结果稳定性）
- 扩展到更多股票和更长时间窗口

---

## 五、已知的工程坑（避免重复踩）

### 坑 1：TradingAgents 需要 Rust 编译器
tiktoken（TradingAgents 的传递依赖）需要从源码编译，必须先安装 Rust 工具链：
```bash
conda install -c conda-forge rust
```

### 坑 2：TradingAgents 记忆文件污染
TradingAgents 会把每次决策的反思写入 `~/.tradingagents/memory/trading_memory.md`。每次开始新的实验前必须清空这个文件，否则之前的记忆会影响新实验的决策。

### 坑 3：DeepSeek 空响应崩溃
原始 TradingAgents 代码在 DeepSeek 返回空字符串时会崩溃整个脚本。已修复：加入3次重试 + 优雅降级（返回 HOLD，confidence=0.0）。相关回归测试：`tests/test_trading_agents_retry.py`。

### 坑 4：Ensemble 零交易问题
"两票一致才行动"的 consensus filter 规则，在 FinRL 极度保守（整个测试期只发一次 BUY）的情况下，会导致 Ensemble 永远不交易。FinRL 保守的根本原因：PPO 在充满矛盾的历史数据（2020-2022 含牛熊）中学会了"少动比多动安全"的局部最优策略。已列为 good-first-issue："设计对信号频率不对称的 agent 鲁棒的 ensemble 规则"。

### 坑 5：Context 过长导致会话中断
长时间运行的 Claude Code 会话会积累大量 token（本项目曾达到 649k uncached），导致 API 错误或上下文压缩丢失记忆。解决方案：使用 tmux 保持进程存活；定期开新会话并用本文件恢复上下文。

### 坑 6：后台进程管理
使用 `nohup + disown` 而不是 Claude Code 的 fork 机制来运行长时间后台任务（已验证在会话中断后进程能存活）：
```bash
nohup conda run -n agentfusion python script.py > /tmp/output.log 2>&1 &
disown
```

---

## 六、项目里程碑状态

```
✔ M0：环境安装 + API 调研（FinRL 和 TradingAgents 均已在各自 conda 环境中真实安装）
✔ v0.1.0：发布到 GitHub（https://github.com/zhuncharlie/agentfusion）
✔ R000-R012：初版实验全部完成（简化版，方法论验证）

◼ M2（进行中）：TradingAgents 真实 adapter（arena/adapters/tradingagents/）
◻ M3：compare.py + 第一份真实对比报告（5股票×60天）
◻ M4：ONBOARDING.md + 下一批候选项目清单
```

---

## 七、下一批候选接入项目

| 项目 | GitHub Stars | 接入难度 | 预估成本/天 | 优先级 |
|------|-------------|---------|-----------|--------|
| Vibe-Trading (HKUDS) | 6.9k | 中 | 待测 | 最高（流量大，社区效应强）|
| AI Hedge Fund (virattt) | 45k+ | 低 | 待测 | 高（star数最多）|
| FinMem | 学术向 | 中 | 低（有记忆层）| 中 |
| FinGPT | 14k | 低 | 极低（情感分析）| 中 |

**接入 Vibe-Trading 前需要确认：**
1. 它是否自己拉数据还是接受外部数据
2. 是否需要预训练
3. 单股还是组合视角
4. 数据格式是否严格

---

## 八、社区运营决策

### good-first-issue 清单（已在 GitHub 上建立）
1. 设计对信号频率不对称的 agent 鲁棒的 ensemble 规则
2. 为 TradingAgents 信号添加 bootstrap 置信区间
3. FinRL 多随机种子训练（验证结果稳定性）
4. 接入 FinMem adapter
5. 扩展股票池到更多标的
6. 替换默认数据源为 Alpha Vantage（解决 yfinance 稳定性风险）
7. 设计声明式配置文件接入方式（v0.2 目标）

### 传播策略（待执行）
- Reddit：`r/algotrading` + `r/MachineLearning`，标题聚焦反直觉发现
- 去 TradingAgents 和 FinRL 仓库发 Discussion，说明集成和发现
- 知乎：面向国内量化社区的中文版介绍
- Hacker News Show HN：等有外部贡献后再发

---

## 九、新会话启动模板

每次开启新 Claude Code 会话时，使用以下 prompt：

```
继续 AgentMesh Arena 项目，路径 /mnt/beegfs/xqinag/ARIS-project。

请先阅读 ARCHITECTURE_DECISIONS.md 了解所有设计背景，再检查当前文件状态，然后从以下断点继续：

[在此处填写当前需要继续的具体任务]

不要重复已完成的工作，不要改变已确认的架构决策。
```

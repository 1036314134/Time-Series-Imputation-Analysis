# Foundational Models 简介

本目录包含多个时间序列基础模型的本地下载脚本与推理封装。下面先给出简短介绍，再用表格汇总关键信息。

## 模型简短介绍

### Chronos-2
Chronos-2 是 Amazon 发布的新一代时序基础模型，采用统一架构处理单变量、多变量及带协变量的预测任务。该模型提供多步分位数预测，适合不确定性评估与区间预测场景；在零样本设定下具有较强泛化能力。当前仓库中的封装接口以两列输入为主，便于与现有实验流程统一。

### Kairos-23M
Kairos-23M 是较轻量的 Kairos 版本，主打自适应 token 化与实例级位置编码，能够在较小参数规模下保持较强零样本预测性能。模型原生输出包含分位数信息，可用于点预测与概率区间估计。当前仓库实现主要按单变量序列调用，适合快速实验、资源受限环境和批量对比评测。

### Kairos-50M
Kairos-50M 是 Kairos 系列的更高容量版本，相比 23M 通常具有更强表示能力与更稳健的跨数据集泛化表现。它同样具备分位数输出能力，适合对预测区间质量有要求的任务。仓库内封装延续统一的单变量输入格式，便于在同一评测脚本下与 Chronos、TimesFM 等模型直接横向比较。

### Moirai-2.0-R-small
Moirai-2.0-R-small 来自 Salesforce 的 Moirai 系列，核心优势在于概率建模与采样式输出，可通过样本分布进一步计算均值、分位数和置信区间。该模型在需要不确定性刻画的场景中较有价值。当前仓库对其做了兼容性封装，支持不同调用分支，并在实验流程中输出统一格式的点预测结果。

### Sundial-base-128m
Sundial-base-128m 是 THUML 推出的生成式时序基础模型，基于自回归生成与采样机制，可一次产生多条未来轨迹，再统计得到均值或分位数。相比仅输出单点的模型，它更适合风险敏感任务。当前仓库封装侧重单变量输入，并加入了与 transformers 版本差异相关的兼容补丁，方便本地稳定推理。

### TimesFM-2.0-500m
TimesFM-2.0-500m 是 Google TimesFM 的第二代开源主力 checkpoint，面向通用单变量预测任务，具备较成熟的工程生态与较强零样本表现。其配置中提供分位数头，但官方说明以点预测为主要目标。仓库内实现兼容 legacy 与 transformers 两种后端，在不同环境下都能较稳定地完成推理。

### TimesFM-2.5-200m
TimesFM-2.5-200m 是 TimesFM 的新版本，特点是更长上下文支持与较新的推理接口，适合长历史窗口场景。模型可同时输出点预测与分位数预测，在长序列任务中实用性较高。当前仓库通过本地 safetensors checkpoint 加载并编译预测配置，沿用统一输入输出格式，便于与 2.0 版本直接对照实验。

### VisionTS++
VisionTS++ 将视觉预训练骨干迁移到时间序列建模，通过跨模态表示学习增强多变量关系建模能力，并提供多分位数预测结果。该思路在复杂相关结构、跨域泛化任务中有潜力。仓库实现会根据 checkpoint 自动识别模型规模并设置预测参数，最终输出与其他模型一致的标准化预测结果，便于统一评估。

## 模型汇总表

说明：  
1. “是否支持多变量”优先按**模型官方能力**整理；当前仓库中的 forecastor 多数按单变量 DataFrame（两列）封装。  
2. “最长上下文长度”以本地 `config.json` / 模型 README 可见信息为准；若未在仓库内明确给出，则标记为“未明确”。

| 模型名字 | checkpoint 网址 | 最长上下文长度 | 是否支持多变量 | 输出格式（分布或分位数） |
|---|---|---:|---|---|
| Chronos-2 | https://huggingface.co/amazon/chronos-2 | 8192 | 是 | 分位数 |
| Kairos-23M | https://huggingface.co/mldi-lab/Kairos_23m | 2048 | 否（当前公开用法以单变量为主） | 分位数 |
| Kairos-50M | https://huggingface.co/mldi-lab/Kairos_50m | 2048 | 否（当前公开用法以单变量为主） | 分位数 |
| Moirai-2.0-R-small | https://huggingface.co/Salesforce/moirai-2.0-R-small | 未明确（本仓库封装为动态 `max(128, 2×forecast_length)`） | 是（模型能力） | 分布（采样） |
| Sundial-base-128m | https://huggingface.co/thuml/sundial-base-128m | 2880 | 未明确（当前封装为单变量） | 分布（采样） |
| TimesFM-2.0-500m | https://huggingface.co/google/timesfm-2.0-500m-pytorch | 2048 | 否（官方说明为 univariate） | 分位数（含点预测） |
| TimesFM-2.5-200m | https://huggingface.co/google/timesfm-2.5-200m-pytorch | 16384 | 否（公开示例以单变量为主） | 分位数（含点预测） |
| VisionTS++ | https://huggingface.co/Lefei/VisionTSpp | 未明确（当前封装 `context_len=len(input)`） | 是（模型能力） | 分位数 |

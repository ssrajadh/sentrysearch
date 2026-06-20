# SentrySearch

**语言 / Languages:** [English](README.md) · 简体中文

对视频素材做语义检索：输入想找的画面描述，自动得到裁剪好的片段。

> [!IMPORTANT]
> **官方来源：** [github.com/ssrajadh/sentrysearch](https://github.com/ssrajadh/sentrysearch) 是 SentrySearch 唯一的官方仓库。其他转载或镜像本项目的网站均与维护者无关，也未获得授权，请始终从本仓库下载。

**工作流程：**
1. **SentrySearch**：在素材里用自然语言或图片找到事件对应片段。
2. **[SentryMerge](https://github.com/ssrajadh/sentrymerge)**：根据搜索结果，把多路摄像头画面自动剪成一条跟随主体的跨镜视频。
3. **[SentryBlur](https://github.com/ssrajadh/sentryblur)**：对敏感区域做本地打码。

**新功能：** [`sentrysearch highlights`](#亮点-highlights)：在还不知道具体搜什么时，按与索引中其他片段的偏离程度对已建索引的切片排序，并自动裁剪导出，适合快速浏览新导入的一批素材。

[<video src="https://github.com/ssrajadh/sentrysearch/raw/main/docs/demo.mp4" controls width="100%"></video>](https://github.com/user-attachments/assets/baf98fad-080b-48e1-97f5-a2db2cbd53f5)

## 目录

- [工作原理](#工作原理)
- [快速开始](#快速开始)
- [用法](#用法)
  - [初始化](#初始化)
  - [建立索引](#建立索引)
  - [搜索](#搜索)
  - [以图搜视频](#以图搜视频)
  - [亮点 Highlights](#亮点-highlights)
  - [Qwen 云端 Alibaba DashScope](#qwen-云端-alibaba-dashscope)
  - [本地后端 无需 API Key](#本地后端-无需-api-key)
  - [本地模型为何能跑得较快](#本地模型为何能跑得较快)
  - [特斯拉元数据叠加层](#特斯拉元数据叠加层)
  - [与 SentryMerge 拼接](#与-sentrymerge-拼接)
  - [使用 SentryBlur 打码](#使用-sentryblur-打码)
  - [管理索引](#管理索引)
  - [详细日志模式](#详细日志模式)
- [技术原理简述](#技术原理简述)
- [费用](#费用)
- [已知告警 可忽略](#已知告警-可忽略)
- [局限与后续方向](#局限与后续方向)
- [兼容性](#兼容性)
- [环境要求](#环境要求)

## 工作原理

SentrySearch 将视频切成有重叠的片段，再分别用 **Google Gemini Embedding API**、阿里云 DashScope（**qwen-cloud**），或本地 **Qwen3-VL** 模型做视频向量化，向量存入本地 **ChromaDB**。搜索时，你的文字（或图片，见 [以图搜视频](#以图搜视频)）被嵌入同一向量空间，与已存视频向量匹配；最佳匹配会从原文件中自动裁切并保存为独立片段。

## 快速开始

1. 安装 [uv](https://docs.astral.sh/uv/)（若尚未安装）：

**macOS / Linux：**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows：**

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. 克隆并安装：

```bash
git clone https://github.com/ssrajadh/sentrysearch.git
cd sentrysearch
uv tool install .
```

> **需要 Python 3.11 或 3.12**（PyTorch 官方轮尚未支持 3.13+）。若本机默认 Python 更新，可先安装托管的 3.12 再固定版本安装：
>
> ```bash
> uv python install 3.12
> uv tool install --python 3.12 .
> ```

3. 配置 API Key（或改用 [本地模型](#本地后端-无需-api-key)）—**仅在使用默认 Gemini 后端时需要**；若全程使用 `--backend local`，或已在 `.env` 中配置 `DASHSCOPE_API_KEY` 并使用 `--backend qwen-cloud`，可跳过本节。

```bash
sentrysearch init
```

会提示输入 Gemini API Key，写入 `.env`，并通过测试嵌入校验。

4. 为素材建索引：

```bash
sentrysearch index /path/to/footage
```

5. 搜索：

```bash
sentrysearch search "red truck running a stop sign"
```

视频切片与裁剪依赖 **ffmpeg**。若系统未安装，会自动使用随包自带的 `imageio-ffmpeg`。

> **手动配置：** 若不想用 `sentrysearch init`，可将 `.env.example` 复制为 `.env`，并手动填入 [aistudio.google.com/apikey](https://aistudio.google.com/apikey) 中的 Key。

## 用法

### 初始化

```bash
$ sentrysearch init
Enter your Gemini API key (get one at https://aistudio.google.com/apikey): ****
Validating API key...
Setup complete. You're ready to go — run `sentrysearch index <directory>` to get started.
```

若已存在 Key，会询问是否覆盖。

> **建议：** 在 [aistudio.google.com/billing](https://aistudio.google.com/billing) 设置消费上限，避免意外超支。

### 建立索引

```bash
$ sentrysearch index /path/to/video/footage
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 1/4]
Indexing file 1/3: front_2024-01-15_14-30.mp4 [chunk 2/4]
...
Indexed 12 new chunks from 3 files. Total: 12 chunks from 3 files.
```

常用选项：

- `--chunk-duration 30` — 每段时长（秒）
- `--overlap 5` — 段与段之间的重叠（秒）
- `--no-preprocess` — 跳过低分辨率/降帧预处理（直接送原始片段）
- `--target-resolution 480` — 预处理目标高度（像素）
- `--target-fps 5` — 预处理目标帧率
- `--no-skip-still` — 即使画面几乎无变化也全部嵌入
- `--backend local` — 使用本地模型而非 Gemini（见下文）

### 搜索

```bash
$ sentrysearch search "red truck running a stop sign"
  #1 [0.87] front_2024-01-15_14-30.mp4 @ 02:15-02:45
  #2 [0.74] left_2024-01-15_14-30.mp4 @ 02:10-02:40
  #3 [0.61] front_2024-01-20_09-15.mp4 @ 00:30-01:00

Saved clip: ./match_front_2024-01-15_14-30_02m15s-02m45s.mp4
```

若最佳结果的相似度低于默认置信阈值（0.41），裁剪前会先询问：

```
No confident match found (best score: 0.28). Show results anyway? [y/N]:
```

使用 `--no-trim` 时，低置信结果仅展示说明，不再交互确认。

其他选项：`--results N`、`--output-dir DIR`、`--no-trim` 跳过自动裁剪、`--threshold 0.5` 调整置信 cutoff、`--save-top N` 保存前 N 个片段而不只最佳。后端与模型一般从索引自动推断；仅在需要覆盖时传 `--backend` 或 `--model`。

### 以图搜视频

用参考图作为查询，适合「找和这张图很像的片段」——用文字不好描述时（例如某辆车的截图、另一段视频的参考帧等）。

```bash
$ sentrysearch img ~/Downloads/image.jpg
  #1 [0.72] 2026-03-12_10-44-17-left_repeater.mp4 @ 00:00-00:30
  #2 [0.69] 2026-03-12_10-44-17-left_repeater.mp4 @ 00:25-00:55
  #3 [0.67] 2026-02-12_20-02-15-front.mp4 @ 00:00-00:18

Saved clip: ./match_2026-03-12_10-44-17-left_repeater_00m00s-00m30s.mp4
```

图片与已索引视频片段在同一向量空间中嵌入，按余弦相似度排序。`search` 的各标志均可用（`--results`、`--threshold`、`--save-top`、`--overlay`、`--no-trim`、`--backend`、`--model`）。

格式支持：Gemini 后端支持 JPG、PNG、WEBP、GIF、HEIC/HEIF；本地后端另支持 PIL 能解码的格式（BMP、TIFF 等）。

> **说明：** 以图搜视频返回的是**视觉相似**的片段，不一定是同一物体。例如用红色轿车查询，可能匹配到其他形状相近的红色轿车——请按场景调整预期。

### 亮点 Highlights

还不知道搜什么？`sentrysearch highlights` 会对索引里「最异常」的切片排序——即 embedding 与整体分布偏离最大的那些——并自动裁剪。适合刚导入一整批素材时先扫一遍。

```bash
$ sentrysearch highlights -n 3
  #1 [0.165] 2026-02-12_20-02-15-back.mp4 @ 00:00-00:18
  #2 [0.163] 2026-02-12_20-02-15-right_repeater.mp4 @ 00:00-00:18
  #3 [0.149] 2026-02-12_20-02-15-front.mp4 @ 00:00-00:18
...
```

评分方式（`--method`）：

- **`knn`**（默认）— 每个切片与其 *k* 个近邻的平均余弦距离；较稳健，能浮出「没有很像孪生」的片段。
- **`centroid`** — 与索引均值的距离；最省算力，偏向「整体里少见」的方向。
- **`lof`** — 局部离群因子（LOF）。索引里存在多种「常态」模式时（白天 / 夜间 / 车库等）更合适。

细化选项：

- `--against "<query>"` — 相对某条查询做异常度。`--against-mode within`（默认）在「该查询的 top 匹配」里再排异常（例如「行人片段里最怪的行人」）；`--against-mode global` 则找「既像该查询、又与索引其余部分都不像」的稀有事件。
- `--dedupe 0.9` — 与更高名次结果过像（默认余弦相似度 0.9）的条目丢弃，避免列表被几乎相同的帧占满。
- `--exclude-baseline` — 先丢掉离质心最近的一半索引再打分；索引被大量重复「无聊」画面主导时有用。
- `-k, --neighbors 10` — `knn` / `lof` 的 *k*。
- `--no-trim` — 只打印排名，不写裁剪文件。

> **注意：** 统计意义上的「异常」不等于「有意思」。传感器 glitch、眩光、夜间帧混在大量白天里、或唯一一段车库录像都会得高分。可配合 `--exclude-baseline`、`--dedupe` 降噪，或用 `--against` 把主题收窄。

### Qwen 云端 Alibaba DashScope

可选用 **qwen-cloud** 后端，对接 [DashScope](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-dashscope) / 模型服务的多模态嵌入（默认模型 `qwen3-vl-embedding`，可用 `--dashscope-model` 或环境变量 `DASHSCOPE_EMBEDDING_MODEL` 覆盖）：

```bash
uv tool install ".[qwen-cloud]"
export DASHSCOPE_API_KEY=...
sentrysearch index /path/to/footage --backend qwen-cloud
sentrysearch search "your query" --backend qwen-cloud
```

**视频上传：** 本地切片文件会先由**官方 Python SDK** 上传到 **DashScope 托管的临时 OSS**，再由 API 消费（HTTP API 需要 URL；SDK 代为完成上传）。

### 本地后端 无需 API Key

使用本地 **Qwen3-VL-Embedding** 建索引与搜索，免费、私密、完全在本地运行。若最看重检索质量，建议用 Gemini 后端；需要离线/隐私时，本地 8B 是稳妥替代；硬件带不动 8B 时可用 2B。

模型会**按硬件自动选择**——NVIDIA GPU 与内存 ≥24GB 的 Mac 倾向 qwen8b；较小内存的 Mac 与纯 CPU 环境倾向 qwen2b。可用 `--model qwen2b` / `--model qwen8b` 覆盖。请按硬件选择安装方式：

| 硬件 | 安装命令 | 自动检测模型 | 说明 |
|---|---|---|---|
| **Apple Silicon，内存 ≥24GB** | `uv tool install ".[local]"` | qwen8b | MPS 上 float16 |
| **Apple Silicon，16GB 内存** | `uv tool install ".[local]"` | qwen2b | 8B 装不下；2B 约 6GB |
| **Apple Silicon，8GB 内存** | `uv tool install ".[local]"` | qwen2b | 较吃紧，高负载可能换页；更建议用 Gemini API |
| **NVIDIA，显存 ≥18GB** | `uv tool install ".[local]"` | qwen8b | bf16（Linux/Windows 会自动拉 CUDA 轮） |
| **NVIDIA，显存 8–16GB** | `uv tool install ".[local-quantized]"` | qwen8b | 4bit 量化（约 6–8GB） |

> **体验会较差：** Intel Mac、无独显的机器会退回到 CPU float32——又慢又占内存，不实用。请改用默认的 **Gemini API 后端**。

> **不确定装哪个？** Mac 用 `".[local]"`；NVIDIA 用 `".[local-quantized]"`——4bit 量化适配面最广、质量损失通常很小。（bitsandbytes 需要 CUDA，不能在 Mac/MPS 上使用。）

**Python 版本：** PyTorch 轮滞后于新 Python，本地后端需要 3.11 或 3.12。若默认已是 3.13+：

```bash
uv python install 3.12
uv tool install --python 3.12 ".[local]"
```

**Mac 前置：** 需安装系统级 FFmpeg（本地模型的视频解码依赖它；Gemini 后端可用自带 ffmpeg）：

```bash
brew install ffmpeg
```

建索引时加 `--backend local`，搜索时一般无需额外参数：

```bash
sentrysearch index /path/to/footage --backend local
sentrysearch search "car running a red light"
```

`search` 会根据索引自动推断后端与模型。也可用 `--model` 作为简写（隐含 `--backend local`）：

```bash
sentrysearch index /path/to/footage --model qwen2b   # 等价于 --backend local --model qwen2b
sentrysearch search "car running a red light"          # 从索引自动识别 local/qwen2b
```

其他选项：

- `--model qwen2b` — 更小模型，质量略低但约 6GB 内存（也支持完整 HuggingFace ID）
- `--quantize` / `--no-quantize` — 强制开启或关闭 4bit 量化（默认：是否安装 bitsandbytes 自动决定）

注意：

- 首次运行会下载模型（8B 约 16GB，2B 约 4GB）。
- **不同后端/模型的向量互不兼容**。每种后端+模型组合有独立索引，不会混用。若用某模型搜索但索引里没有对应数据，会提示实际使用的模型。
- 速度与 GPU 核心数等相关——基础款 M 系列会比 Pro/Max 慢，但结果一致。

### 本地模型为何能跑得较快

本地后端通过几项叠加策略控制速度与内存：

- **预处理在进入模型前缩小片段。** 每段约 30 秒的片段会先经 ffmpeg 降到 480p、5fps。例如约 19MB 的行车记录仪片段可缩到约 1MB——模型要处理的像素量大幅下降。推理耗时与像素量相关，而非单纯与视频时长成正比，因此这是最大的加速来源之一。
- **低帧采样。** 视频处理器每段最多送 32 帧（`fps=1.0`，`max_frames=32`）。30 秒片段约 30 帧，而不是成百上千帧。
- **MRL 维度截断。** Qwen3-VL-Embedding 支持 [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147)。只保留每条嵌入的前 768 维并做 L2 归一化，减轻 ChromaDB 存储与距离计算。
- **自动量化。** 在显存有限的 NVIDIA 上，8B 会自动以 4bit（bitsandbytes）加载——从约 18GB 降到约 6–8GB，质量损失通常很小。4090（24GB）可舒适跑 bf16 全精度。
- **静止画面跳过。** 通过比较采样帧的 JPEG 体积判断画面是否几乎无变化，无变化则跳过整段嵌入——每段省一次完整前向。

综合以上，A100 上约每段 2–5 秒，T4 约 3–8 秒；4090 上 8B bf16 通常每段个位数秒级。

### 特斯拉元数据叠加层

在裁剪出的片段上叠加速度、位置与时间：

```bash
sentrysearch search "car cutting me off" --overlay
```

会从特斯拉行车记录仪文件中读取内嵌遥测（速度、GPS）并渲染 HUD。叠加层包含：

- **顶部居中：** 速度与 MPH 标签，浅灰底卡片
- **卡片下方：** 日期与时间（12 小时制，含 AM/PM）
- **左上角：** 城市与道路名（反向地理编码）

![特斯拉叠加层](docs/tesla-overlay.png)

要求：

- 特斯拉车机固件 2025.44.25 及以上，HW3+
- SEI 元数据仅出现在行驶录像中（不含驻车/哨兵模式）
- 反向地理编码通过 geopy 调用 [OpenStreetMap Nominatim API](https://nominatim.openstreetmap.org/)（可选）

带特斯拉叠加层支持的安装：

```bash
uv tool install ".[tesla]"
```

未安装 geopy 时叠加层仍可用，但不会显示城市/路名。

来源：[teslamotors/dashcam](https://github.com/teslamotors/dashcam)

### 与 SentryMerge 拼接

[SentryMerge](https://github.com/ssrajadh/sentrymerge) 是配套工具：在拿到 SentrySearch 的搜索结果后，自动从多路摄像头素材中裁出**同一场景**的片段，借助 VLM 标定各路画面的起止时间（精确到子秒），再拼成**一条**镜头随主体移动的跨镜视频。每次运行 `sentrysearch search` 会把结果列表写入 `~/.sentrysearch/last_search.json`，SentryMerge 通过 `--last` 读取，无需手传路径：

```bash
sentrysearch search "<查询>"
sentrymerge --last                      # → merge.mp4
```

不必重新搜索也可直接用 `--last`；`sentrymerge --query "..."` 会在内部重新跑一遍搜索。安装说明、VLM 后端（Gemini / OpenAI / 本地 Qwen）以及非特斯拉行车记录仪的模块化相机配置，见 [SentryMerge README](https://github.com/ssrajadh/sentrymerge#readme)。

### 使用 SentryBlur 打码

[SentryBlur](https://github.com/ssrajadh/sentryblur) 是配套工具，可在本地对视频做人脸、车牌与自然语言指定的区域打码。每次 `sentrysearch search` 保存片段后，会把路径缓存在 `~/.sentrysearch/last_clip.json`；SentryBlur 通过 `--last` 读取，因此「先搜再打码」只需两条命令、无需手传路径：

```bash
sentrysearch search "car cuts me off"
sentryblur prompt --last "road signs"   # → match_<...>_blurred.mp4
```

`sentryblur faces --last` 与 `sentryblur plates --last` 同理。`faces` / `plates` 走较快 CPU 检测器；`prompt "<text>"` 可指定任意物体（手机屏幕、显示器、工牌等）——`prompt` 需要 NVIDIA GPU 或 Apple Silicon。安装与硬件说明见 [SentryBlur README](https://github.com/ssrajadh/sentryblur#readme)。

### 管理索引

```bash
# 查看索引信息（标记为 [missing] 表示磁盘上已不存在）
sentrysearch stats

# 按路径子串删除指定文件
sentrysearch remove path/to/footage

# 清空整个索引
sentrysearch reset
```

### 详细日志模式

在 `index` 或 `search` 上加 `--verbose`，可输出调试信息（嵌入维度、API 耗时、相似度分数等）。

## 技术原理简述

Gemini Embedding 2 与 Qwen3-VL-Embedding 都能**原生嵌入视频**——原始视频像素与文字查询被映射到同一向量空间。没有转写、没有逐帧生成字幕、没有「文字中间层」。像「红灯前的一辆红卡车」这样的查询，可以直接与 30 秒视频片段在向量层面比较，这才让「在数小时素材里做近实时语义检索」变得可行。

## 费用

### Gemini

在默认设置（30 秒片段、5 秒重叠）下，用 Gemini 嵌入 API 索引 **约 1 小时** 素材大约 **2.84 美元**：

> 1 小时 = 3600 秒视频 = 模型处理 3600 帧。  
> 3600 帧 × $0.00079 ≈ **$2.84 / 小时**

Gemini API 对上传视频按 **每秒 1 帧** 抽取并计费，与文件原始帧率无关。本地的预处理（ffmpeg 降到 480p、5fps）主要减小体积与传输时间、降低超时风险，**不改变** API 实际计费的帧数。

两项内置优化从不同角度降低成本：

- **预处理（默认开启）** — 上传前降到 480p、5fps。因 API 仍按 1fps 计费，这主要加快上传、避免超时，**不减少**计费帧数。
- **静止跳过（默认开启）** — 几乎无变化的片段整段跳过，**直接少调用 API**，省钱幅度取决于素材：哨兵模式长时间静止收益大；全程激烈驾驶的素材可能几乎无可跳片段。

搜索侧费用可忽略（多为文本嵌入）。

### Qwen 云端 DashScope

DashScope 对**多模态向量**按 **人民币 / 千输入 Token**、按**模态**计费。默认模型 `qwen3-vl-embedding` 在阿里云公开文档中的参考单价（**以你账号地域、控制台与文档最新价为准**）一般为：

- **文本输入：**约 **¥0.0007 / 千 Token**  
- **图片 / 视频输入：**约 **¥0.0018 / 千 Token**  

建索引时每个切片走 **视频** 模态；`search` / `img` 查询主要是**文本**或**图片** Token，单价通常低于视频。总费用取决于每次请求在账单里统计的 **Token 数**（与分辨率、时长、采样如环境变量 `DASHSCOPE_VIDEO_FPS` 等有关），**不能像 Gemini 那样仅凭「每秒一帧」的公开美元单价直接换算成固定「每小时多少钱」**，建议先小规模索引后在百炼/ DashScope 控制台看实际消耗再外推。

文档中常见**新开通限时免费 Token 额度**（例如百万级 Token、限期有效），以 [多模态向量计费说明](https://help.aliyun.com/dashscope/developer-reference/one-peace-multimodal-embedding-metering-and-billing) 与控制台为准，**定价与活动会调整**。

### 索引进阶选项 两种后端

以下参数同时影响 **Gemini** 与 **qwen-cloud** 的切片与预处理：

- `--chunk-duration` / `--overlap` — 片段更长、重叠更少 → API 调用更少 → 成本更低
- `--no-skip-still` — 即使画面静止也全部嵌入
- `--target-resolution` / `--target-fps` — 调整预处理画质
- `--no-preprocess` — 直接上传未预处理片段

## 已知告警 可忽略

本地后端在索引/搜索时可能打印告警，多为外观提示，**不影响结果**：

- **`MPS: nonzero op is not natively supported`** — Apple Silicon 上 PyTorch 的已知限制，该算子一步会回退 CPU，其余仍在 GPU。对输出质量无影响。
- **`video_reader_backend torchcodec error, use torchvision as default`** — macOS 上 torchcodec 找不到兼容 FFmpeg，视频处理器自动回退 torchvision，属预期行为，结果一致。
- **`You are sending unauthenticated requests to the HF Hub`** — 从 Hugging Face 下载模型时未带 Token，速度可能略慢，模型仍可正常加载。若介意可设置环境变量 `HF_TOKEN`。

## 局限与后续方向

- **静止检测为启发式** — 基于采样帧 JPEG 体积比较，可能偶尔漏掉有细微运动的片段，或给真正静止的片段建索引。若必须逐段索引，请使用 `--no-skip-still`。
- **检索质量受切片边界影响** — 若事件跨两段，重叠窗口有帮助但不完美。更智能的切片（如场景检测）或可改善。
- **Gemini Embedding 2 处于预览阶段** — API 行为与定价可能变更。

## 兼容性

支持 `.mp4` 与 `.mov`，不限于特斯拉哨兵模式。目录扫描会递归查找这两种扩展名，与文件夹结构无关。

## 环境要求

- Python 3.11+
- 系统 `ffmpeg` 在 PATH 中，或默认安装带来的 `imageio-ffmpeg` 捆绑 ffmpeg
- **Gemini 后端：** Gemini API Key（[免费申请](https://aistudio.google.com/apikey)）
- **本地后端：**
  - 支持 CUDA 的 GPU 或 Apple Metal（显存/内存要求见 [上文表格](#本地后端-无需-api-key)）
  - **macOS：** `brew install ffmpeg`（视频解码需要）
  - **Linux / Windows：** 无额外系统依赖

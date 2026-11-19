# 焊缝外观缺陷检测（weld-defect-detector）

一个开箱即用的焊缝外观缺陷检测工具：输入一张焊缝照片，自动框出缺陷并分类，输出画框图和结构化 JSON。

基于 [RH821/ship-ai-system](https://github.com/RH821/ship-ai-system) 的焊缝外观模块重构而来，原算法逻辑保持不变，仅做了工程化清理（移除硬编码路径与未解决的合并冲突、参数化输入输出）。原项目代码为 MIT 许可，模型权重的许可请见文末说明。

## ☁️ 最快上手：GitHub 云端运行（完全不用装环境，推荐）

**不用装 Python、不用装 PyTorch、不用敲任何命令**——全在 GitHub 服务器上跑，跑完点开网页看结果。

1. 打开仓库页面，进入 `我的照片` 文件夹，点 `Add file` → `Upload files`，上传**脱敏后**的焊缝照片（船名/分段号/日期水印先裁掉），提交。
2. 上传后云端会**自动开始检测**（也可到 `Actions` 页手动点 `Run workflow`，能调参数或选内置示例）。
3. 等 3–8 分钟，回到仓库 `results` 文件夹，点开 **`报告.md`** —— 每张图的画框结果、缺陷类别、置信度、坐标全在里面，图片直接显示。
4. 同样在 `Actions` 页该次运行底部可下载 `检测结果` 压缩包（含全部结果图和 JSON）。

> 想先看看效果、不传自己的照片：到 `Actions` → `云端焊缝检测` → `Run workflow` → 模式选 `sample` → 运行，看 `results/报告.md` 即可。
>
> ⚠️ 云端方案会把照片上传到 GitHub。仓库是公开的，**务必只传脱敏照片**；若要传含敏感信息的图，请先把仓库改成 Private（`Settings` → `Change visibility`），Actions 私有仓库也有免费额度。

## 最快上手：本地双击运行（Windows，自己有 Python 环境时）

1. 确认电脑装了 **Python 3.10 或 3.11**（安装时勾选 "Add Python to PATH"）
2. 把整个项目文件夹下载/解压到本地
3. **双击 `一键运行.bat`** —— 它会自动创建环境、装依赖、跑示例、打开结果文件夹
4. 想检测自己的照片：把**脱敏后**的照片放进自动生成的 `我的照片` 文件夹，双击 `检测我的照片.bat`

首次运行要下载 PyTorch 等依赖，约 5–15 分钟，之后秒开。想手动控制参数就看下面的命令行用法。

## 能检测什么

对标船舶焊缝质量国家标准 **CB/T 3802-2019** 的五类外观缺陷：

| 类别名 | 中文 | 画框颜色 |
|---|---|---|
| `undercut` | 咬边 | 红 |
| `slaginclusion` | 夹渣 | 绿 |
| `porosity` | 气孔 | 蓝 |
| `crack` | 裂纹 | 青 |
| `overlap` | 未熔合 | 橙 |

原项目公开的指标：mAP 72.5%，FPS 67，训练数据为华南某大型船厂小组立车间实拍焊缝图。**这些是作者在其数据集上的结果，不代表在任意现场照片上的表现**——换光照、焊接方法、拍摄角度都可能掉点，这正是需要你用自己的脱敏照片去验证的地方。

## 工作原理

两段式检测，不是直接一把梭：

1. **第一阶段 · 焊缝区域定位**：`weld_area.pt` 在整图里框出焊缝条带（取面积最大的框），先把无关背景排除掉。没检到焊缝会直接输出空结果，不会硬报缺陷。
2. **第二阶段 · 缺陷检测**：把焊缝条带旋转成横向、统一缩放到高 640，再切成 640×640 的小瓦片逐块送 `liangdutongyi.pt` 检测（小目标更清楚）。检测前做 LAB 空间 CLAHE + 高光压缩 + 暗部提升，专门对付现场"强反光 + 局部暗"。
3. **坐标还原**：把瓦片里的缺陷框换算回原图坐标，在整图上画框并输出 JSON。

## 安装

环境要求：**Python 3.9+**（有 NVIDIA 显卡会快很多，没有也能跑 CPU）。

```bash
# 1. 进入项目目录
cd weld-defect-detector

# 2. 建虚拟环境（推荐，避免污染系统 Python）
python -m venv .venv
# Windows 激活：
.venv\Scripts\activate
# macOS / Linux 激活：
# source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> 第一次安装会拉取 PyTorch，体积较大（CPU 版约 200MB+，GPU 版按 CUDA 版本更大）。如果已有 PyTorch 环境，可跳过重复安装。

## 运行

### 跑通内置示例（零配置）

仓库自带一张示例焊缝图和它的预期结果图，先跑它确认环境没问题：

```bash
python detect.py -i samples/demo.jpg -o outputs
```

跑完看 `outputs/` 目录：

- `demo_vis.jpg` —— 画框结果图（可与 `samples/demo_expected_result.jpg` 对比）
- `demo_result.json` —— 每个缺陷的类别、置信度、原图坐标
- `summary.json` —— 本次批量处理的汇总

### 跑你自己的照片

**方式一：单张图**

```bash
python detect.py -i "D:/myphotos/weld_001.jpg" -o outputs
```

**方式二：整个文件夹（批量）**

把脱敏后的照片放进一个文件夹，直接指目录：

```bash
python detect.py -i "D:/myphotos" -o outputs
```

### 常用参数

```bash
python detect.py -i samples/demo.jpg -o outputs \
    --conf 0.25 \
    --overrate 0.2 \
    --save-tiles \
    --device cpu
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `-i / --input` | `samples` | 输入：单张图片路径或图片目录 |
| `-o / --output` | `outputs` | 结果输出目录 |
| `-w / --weights-dir` | `./weights` | 权重目录 |
| `--conf` | `0.25` | 置信度阈值。调低→检出更多但误报增多，调高→更保守 |
| `--overrate` | `0.0` | 瓦片横向重叠率 0~0.9。长焊缝怕漏检可调到 `0.2` |
| `--no-preprocess` | 关 | 关闭亮度归一化。若你的照片光线很好、预处理反而 distort，可试 |
| `--save-tiles` | 关 | 保存中间瓦片图到 `outputs/_tiles/`，用于排查"为什么这张没检出" |
| `--no-visual` | 关 | 只输出 JSON，不生成画框图 |
| `--device` | 自动 | 强制 `cpu` / `cuda` / `0`（0 号显卡） |

跑 `python detect.py --help` 看完整参数。

## 结果 JSON 长什么样

```json
{
  "task": "weld_defect",
  "source": "samples/demo.jpg",
  "timestamp": "2026-11-19T13:50:00",
  "params": { "conf": 0.25, "overrate": 0.0, "normalize_tiles": true },
  "result": {
    "image_size": [1280, 720],
    "n_detections": 2,
    "detections": [
      {
        "bbox_xyxy": [312.5, 188.0, 356.2, 220.7],
        "category": "porosity",
        "category_zh": "气孔",
        "score": 0.83
      }
    ]
  }
}
```

`bbox_xyxy` 是原图坐标系的左上角 + 右下角，可直接对接后续统计（缺陷计数、面积估算、人工复核清单）。

## 现场使用的三点提醒

1. **先脱敏再喂图**。照片里若含船名、分段号、船东标识、日期水印，务必先裁掉或打码。本工具只在本机推理、不联网上传图片，但从你电脑发出去对比、汇报之前要脱敏——这是红线。
2. **AI 只做初筛，人工复核确认**。这个模型的定位是"帮质检员先看一遍、减少漏看"，不是替代质检签字。受控的质量记录仍以人工确认为准。
3. **别用公开指标当承诺**。72.5% 的 mAP 是作者数据集上的数字。在你们厂的光照和焊接方法下，第一次跑可能明显低于它——这很正常，记录下来就是后续微调的起点。

## 目录结构

```
weld-defect-detector/
├── detect.py                  # 主程序（命令行入口，本地/云端共用）
├── make_report.py             # 云端报告生成器（JSON → 网页报告）
├── .github/workflows/
│   └── cloud-detect.yml       # 云端自动检测工作流
├── 我的照片/                  # 把脱敏照片传到这里，自动触发云端检测
├── samples/                   # 示例图 + 预期结果图
├── weights/                   # 三个模型权重
│   ├── weld_area.pt           # 阶段一：焊缝区域定位（5.4MB）
│   ├── liangdutongyi.pt       # 阶段二：缺陷检测（19MB）
│   └── best.pt                # 原项目附带权重（本脚本未用到，保留备查）
├── outputs/                   # 本地运行结果
└── results/                   # 云端运行结果（含 报告.md，自动提交回仓库）
```

## 关于权重与许可

- **代码**：基于原项目 MIT 许可重构，可自由使用与修改。
- **模型权重**：原项目 README 注明"模型权重与数据集受各自协议约束"，但未明确具体条款；权重由原作者在其船厂数据上训练。因此**本仓库的权重仅供个人学习与原型验证使用**，商用前请联系原作者确认授权。
- 若权重下载链接失效，原项目地址：https://github.com/RH821/ship-ai-system

## 常见问题

**Q：提示 `No module named ultralytics`？**
A：虚拟环境没激活，或没装依赖。回到安装步骤执行 `pip install -r requirements.txt`。

**Q：CPU 跑一张图很慢？**
A：正常，两段式 + 切片推理在 CPU 上可能要几十秒。把照片分辨率降到合适尺寸、或 `--overrate 0.0` 会快很多。有显卡会自动用。

**Q：`未检测到焊缝区域` 是坏了？**
A：不是。说明第一阶段的定位模型没在这张图里找到焊缝——常见于照片太远、焊缝占比太小、或角度太偏。换张焊缝更清晰的近景图试试，或看 `outputs/_tiles/` 里的中间图排查。

**Q：想用它做专利/论文？**
A：核心创新点在原项目的 WCM 小波卷积模块和 SIoU 损失（见原项目 `modules/weld_defect/WTconv/`），本工具用的是训练好的权重。若要研究模型结构本身，请回原项目看源码。

# PDF2MD - PDF to Markdown Converter

不使用 OCR 和 LLM 的轻量化 PDF 转 Markdown 工具，支持图片提取、表格识别、链接提取和标题自动识别。

> **⚠️ 使用限制**
> - **内容要求**：对数学符号公式等束手无策
> - **布局限制**：目前仅支持单栏布局的 PDF 转换
> - **PDF 类型**：仅支持文本型 PDF（扫描版 PDF 暂不支持）
> - **页数建议**：建议 PDF 页数不超过 2000 页，避免显存占用过高
> - **质量要求**：PDF 本身质量越好，转换效果越好

## 功能特性

- [x] **智能文本提取**：自动提取 PDF 中的文本内容
- [x] **图片检测与提取**：基于 YOLO 模型的文档布局检测，自动识别并提取图片
- [x] **表格识别**：自动识别表格并转换为 Markdown 格式
- [x] **链接提取**：保留 PDF 中的超链接
- [x] **大纲识别**：自动识别 PDF 大纲结构并生成标题层级
- [x] **页眉页脚过滤**：自动过滤并移除页眉页脚内容
- [x] **灵活配置**：支持 JSON/YAML/INI 多种配置文件格式
- [ ] **智能排版**：多栏布局检测（待完善）

## 项目结构

```
pdf2md/
├── core/                    # 核心功能模块
│   ├── extractor.py         # 图片和文本提取器
│   └── schema.py            # 数据模型定义
├── utils/                   # 工具函数
│   └── utils.py             # 配置加载、目录准备等工具
├── config/                  # 配置文件
│   └── config.ini           # 默认配置文件
├── model/                   # 模型文件
│   └── doclayout_yolo/      # YOLO 文档布局检测模型
├── pdf_converter.py         # 主转换器类
└── README.md
```

## 安装

### 1. 克隆仓库

```bash
git clone <repository-url>
cd pdf2md
```

### 2. 安装依赖

```bash
pip install pdfplumber pandas torch pypdf pydantic numpy tqdm doclayout-yolo pyyaml
```

## 快速开始

### 基本使用

```python
from pdf_converter import PDFConverter

# 从配置文件创建转换器
converter = PDFConverter.create_converter("config/config.ini")

# 转换 PDF 到 Markdown
markdown_content = converter.convert_to_markdown(
    input_file_path="data/pdfs/example.pdf",
    output_file_name="example.md",  # 可选：指定输出文件名
    output_folder="output/"         # 可选：指定输出文件夹
)

print(markdown_content)
```

### 直接使用配置字典

```python
from pdf_converter import PDFConverter
from utils import load_config

# 手动加载配置
config = load_config("config/config.ini")  # 支持 .json, .yaml, .ini

# 创建转换器
converter = PDFConverter(config)

# 转换
markdown_content = converter.convert_to_markdown("path/to/file.pdf")
```

## 配置说明

配置文件支持三种格式：JSON、YAML 和 INI。默认使用 `config/config.ini`。

### 配置文件结构

```ini
[doclayout_yolo]
model_path=./model/doclayout_yolo/doclayout_yolo_docstructbench_imgsz1024.pt

[image_extractor]
# YOLO 模型检测参数
yolo_predict_resolution=180      # 预测时图片分辨率
yolo_imgsz=1024                  # YOLO 输入尺寸
yolo_conf=0.65                   # 置信度阈值
yolo_image_class_id=3             # 图片类别索引
yolo_verbose=false                # 是否显示详细信息

# 图片处理和保存参数
save_dpi=720                      # 保存图片的 DPI
overlap_thresh=0.75               # 重叠框合并阈值
transfer_dpi=180                  # 坐标转换 DPI

[text_extractor]
# 文字提取容差参数
x_tolerance=1                     # X 方向容差
y_tolerance=1                     # Y 方向容差
keep_blank_chars=false            # 是否保留空白字符
```

### 配置参数说明

#### 图像提取器 (image_extractor)

- `yolo_predict_resolution`: 用于 YOLO 检测的图片分辨率（默认 180）
- `yolo_imgsz`: YOLO 模型输入图片尺寸（默认 1024）
- `yolo_conf`: 检测置信度阈值，范围 0-1（默认 0.65）
- `yolo_image_class_id`: 图片在 YOLO 模型中的类别 ID（默认 3）
- `save_dpi`: 保存提取图片的 DPI，值越高图片质量越好但文件越大（默认 720）
- `overlap_thresh`: 当两个检测框重叠度超过此值时会被合并（默认 0.75）
- `transfer_dpi`: 用于坐标转换的 DPI（默认 180）

#### 文本提取器 (text_extractor)

- `x_tolerance`: 提取文字时 X 方向的容差像素（默认 1）
- `y_tolerance`: 提取文字时 Y 方向的容差像素（默认 1）
- `keep_blank_chars`: 是否保留空白字符（默认 false）

## 使用示例

### 示例 1：基本转换

```python
from pdf_converter import PDFConverter

converter = PDFConverter.create_converter("config/config.ini")
result = converter.convert_to_markdown("document.pdf")
```

### 示例 2：指定输出路径

```python
converter = PDFConverter.create_converter("config/config.ini")
result = converter.convert_to_markdown(
    input_file_path="document.pdf",
    output_file_name="converted_document.md",
    output_folder="output/"
)
```

### 示例 3：使用测试脚本

```python
python test/test_pdf_converterm.py
```

### 示例 4：查看 PDF 大纲

```bash
python scripts/view_outline.py path/to/document.pdf
```

## 输出结构

转换后的输出目录结构：

```
output_directory/
├── document.md              # Markdown 主文件
└── images/                  # 提取的图片
    ├── page_0_image_0.png
    ├── page_0_image_1.png
    └── ...
```

## 核心功能说明

### 1. 图片提取

使用 YOLO 文档布局检测模型自动识别 PDF 中的图片位置，支持：
- 自动检测图片边界
- 高分辨率图片保存（可配置 DPI）
- 重叠检测框合并
- 相对路径图片链接生成

### 2. 文本提取

- 自动识别标题（基于 PDF 大纲或正则规则）
- 表格自动转换为 Markdown 表格
- 链接转换为 Markdown 链接格式
- 自动过滤页眉页脚

### 3. 大纲识别

- 如果 PDF 包含大纲（outline），自动识别标题层级
- 根据标题相似度匹配页面元素
- 自动生成 Markdown 标题（`#`, `##`, `###` 等）

### 4. 多栏检测（待完善）

目前仅支持检测 PDF 是否使用多栏布局，并给出警告提示。多栏布局的准确转换功能正在开发中。

## 依赖库

- `pdfplumber`: PDF 文本和表格提取
- `pypdf`: PDF 大纲解析
- `doclayout-yolo`: 文档布局检测模型
- `torch`: 深度学习框架
- `pandas`: 表格处理
- `pydantic`: 数据验证
- `numpy`: 数值计算
- `tqdm`: 进度条显示
- `pyyaml`: YAML 配置文件支持（可选）

## 开发

### 运行测试

```bash
python test/test_pdf_converterm.py
```

### 项目模块说明

- `core/extractor.py`: 图片和文本提取器实现
- `core/schema.py`: PDFFile、PDFPage、PageElement 等数据模型
- `utils/utils.py`: 配置加载、目录准备等工具函数
- `pdf_converter.py`: 主转换器类

## 注意事项

1. **模型文件**：首次使用需要下载 YOLO 模型文件到 `model/doclayout_yolo/` 目录
2. **GPU 支持**：如果有 CUDA 支持的 GPU，程序会自动使用 GPU 加速图片检测
3. **内存占用**：处理大型 PDF 文件时可能占用较多内存，建议在配置充足的环境中运行
4. **图片质量**：`save_dpi` 参数影响输出图片质量，值越大文件越大

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 更新日志

### v0.1.0
- 初始版本
- 支持 PDF 转 Markdown 基本功能
- 图片提取（YOLO 模型）
- 文本、表格、链接提取
- 大纲识别和标题自动生成
- 支持多种配置文件格式


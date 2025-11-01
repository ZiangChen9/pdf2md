from __future__ import annotations

import os
from typing import Any, Dict, Optional

import pdfplumber
from tqdm import tqdm

from core.extractor import ImageExtractor, TextExtractor
from core.schema import PDFFile, PDFPage
from utils import load_config, prepare_directories


class PDFConverter:
    """PDF to Markdown converter"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @classmethod
    def create_converter(cls, config_path: str) -> PDFConverter:
        config = load_config(config_path)
        return cls(config)

    def convert_to_markdown(
        self,
        input_file_path: str,
        output_file_name: Optional[str] = None,
        output_folder: Optional[str] = None,
    ) -> str:
        # 初始化一个pdffile对象，如果PDF有outline属性则会自动获取
        pdffile = PDFFile(input_file_path, debug=False)
        pdffile.check_multicolumn()  # 抛出多栏告警
        # 准备工作目录，若未提供图片输出目录和输出名称，则将在pdf的同级目录下创建和pdf同名的工作目录
        output_markdown_dir, image_output_dir = prepare_directories(
            input_file_path, output_file_name, output_folder
        )
        # 移除工作残留
        if os.path.exists(output_markdown_dir):
            os.remove(output_markdown_dir)
        # 图片和文字提取器，如果outline不可用则会开启标题的正则化判断
        image_extractor = ImageExtractor(
            self.config, image_output_dir, output_markdown_dir
        )
        # 标题不可用则要开启标题预测
        text_extractor = TextExtractor(
            config=self.config, header_predict_work=(not pdffile.can_use_outlines)
        )
        # 主循环
        with pdfplumber.open(input_file_path) as pdf:
            for i in tqdm(range(len(pdf.pages)), desc="Converting pages", unit="page"):
                p = pdf.pages[i]  # 获取pdf的一页
                # 使用配置中的默认参数，可以在调用时覆盖
                images = image_extractor.execute(p, page_index=i)
                texts = text_extractor.execute(p, page_index=i)
                elements = images + texts  # 获取所有元素
                # 构建一个page对象
                page = PDFPage(page_idx=i, page_size=(p.width, p.height))
                # 将elements加到page时会自动清除页眉和页脚，并且当页元素位置会自动就绪
                page.add_filter_sort_elements(elements)
                # 添加page到pdffile,如果有header则会refine
                pdffile.add_and_refine_page(page)
        markdown_content = pdffile.element_to_markdown(
            output_file_path=output_markdown_dir
        )
        return markdown_content

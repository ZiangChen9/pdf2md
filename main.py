import gc
import os
import warnings
import pdfplumber
from tqdm import tqdm

from .schema import PDFFile, PDFPage
from .extractor import ImageExtractor, TextExtractor


class PDFConverter:
    def __init__(self, config: dict):
        self.config = config
        self.llm_client = ClientCreator(config).create_llm_client()
        self.llm_model_name = config["llm"]["general_model_name"]

    def convert_to_markdown(
        self,
        input_file_path: str,
        output_file_name: str = None,
        output_folder: str = None,
    ) -> str:
        # 初始化一个pdffile对象，如果PDF有outline属性则会自动获取
        pdffile = PDFFile(input_file_path, debug=False)
        pdffile.check_multicolumn()  # 抛出多栏告警
        # 准备工作目录，若未提供图片输出目录和输出名称，则将在pdf的同级目录下创建和pdf同名的工作目录
        output_markdown_dir, image_output_dir = U.prepare_directories(
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
            header_predict_work=(not pdffile.can_use_outlines)
        )
        # 主循环
        with pdfplumber.open(input_file_path) as pdf:
            for i in tqdm(range(len(pdf.pages)), desc="Converting pages", unit="page"):
                p = pdf.pages[i]  # 获取pdf的一页
                images = image_extractor.execute(
                    p, page_index=i, save_dpi=720, overlap_thresh=0.75
                )
                texts = text_extractor.execute(p, page_index=i)
                elements = images + texts  # 获取所有元素
                # 构建一个page对象
                page = PDFPage(page_idx=i, page_size=(p.width, p.height))
                # 将elements加到page时会自动清除页眉和页脚，并且当页元素位置会自动就绪
                page.add_filter_sort_elements(elements)
                # 添加page到pdffile,如果有header则会refine
                pdffile.add_and_refine_page(page)
        # 手动调用一下缓存清理，主要是回收cuda的显存占用
        image_extractor.clear_cache()
        text_extractor.clear_cache()
        del image_extractor, text_extractor
        gc.collect()
        # 如果大纲不可用
        if not pdffile.can_use_outlines:
            try:
                # 尝试创建大模型客户端并修复标题
                llm_client = ClientCreator(self.config).create_llm_client()
                llm_model_name = self.config["llm"]["general_model_name"]
            except Exception as e:
                warnings.warn(
                    f"Fail to create llm client: {e}",
                    UserWarning,
                )
                markdown_content = pdffile.element_to_markdown(
                    output_file_path=output_markdown_dir
                )
                return markdown_content
            pdffile.refine(llm_client=llm_client, llm_model_name=llm_model_name)
        markdown_content = pdffile.element_to_markdown(
            output_file_path=output_markdown_dir
        )
        return markdown_content

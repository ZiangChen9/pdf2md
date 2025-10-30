import difflib
import json
import random
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Deque, Dict, List, Literal, Tuple, TypeVar, Union

import numpy as np
import pdfplumber
from pydantic import BaseModel
from pypdf import PdfReader
from pypdf.generic import Destination

T = TypeVar("T", float, int)
BoundingBox = Tuple[T, T, T, T]
ElementType = Literal["plain_text", "table", "link", "figure", "header"]


@dataclass
class OutlineInfo:
    """Outline Message in PDF"""

    level: int  # Outline level
    title: str  # Outline title
    page_number: int  # Outline page number


class PageElement(BaseModel):
    """Page Element in PDF"""

    content_type: ElementType  # Element type
    element_id: int  # Element unique id
    page_idx: int  # Element page index
    content: str  # Element content
    bbox: BoundingBox  # Element bounding box
    _id_counter: ClassVar[int] = 0  # Ensure unique id

    def __init__(self, **data):
        if "element_id" not in data or data["element_id"] is None:
            type(self)._id_counter += 1
            data["element_id"] = type(self)._id_counter
        super().__init__(**data)

    def to_dict(self):
        return dict(
            type=self.content_type,
            element_id=self.element_id,
            page_idx=self.page_idx,
            content=self.content,
            bbox=self.bbox,
        )


class PDFPage:
    """
    Single page: collect and sort page elements, provide filtering header and footer functionality
    """

    def __init__(self, page_idx: int, page_size: Tuple[float, float]):
        self.page_idx = page_idx
        self.page_size = page_size  # （宽，高）
        self.elements: List[PageElement] = []

    def add_and_sort_element(self, element: PageElement):
        """
        添加元素，过滤页眉页脚并保持阅读顺序
        """
        if element.page_idx != self.page_idx:
            raise ValueError(
                f"Element page_idx ({element.page_idx}) "
                f"does not match OnePage page_idx ({self.page_idx})"
            )
        self.elements.append(element)
        self._filter_header_footer()
        self._sort()

    def add_filter_sort_elements(self, elements: List[PageElement]):
        """
        一次性添加多个元素
        """
        for el in elements:
            if el.page_idx != self.page_idx:
                raise ValueError(
                    f"Element page_idx ({el.page_idx}) "
                    f"does not match OnePage page_idx ({self.page_idx})"
                )
        self.elements.extend(elements)
        self._filter_header_footer()
        self._sort()

    def _sort(self):
        """Sort elements in reading order for single column"""

        def line_y(el: PageElement) -> float:
            _, y0, _, y1 = el.bbox
            return (y0 + y1) / 2

        def col_x(el: PageElement) -> float:
            x0, _, x1, _ = el.bbox
            return (x0 + x1) / 2

        self.elements.sort(key=lambda e: (line_y(e), col_x(e)))

    def _filter_header_footer(self, margin_ratio: float = 0.02):
        """Filter header and footer"""
        kept_elements = []
        page_width, page_height = self.page_size
        margin_h = page_height * margin_ratio
        margin_w = page_width * margin_ratio

        for element in self.elements:
            if len(element.bbox) != 4:
                continue
            x0, y0, x1, y1 = element.bbox
            is_in_header = y0 < margin_h
            is_in_footer = y1 > page_height - margin_h
            is_in_left_margin = x0 < margin_w
            is_in_right_margin = x1 > page_width - margin_w
            if not (
                is_in_header or is_in_footer or is_in_left_margin or is_in_right_margin
            ):
                kept_elements.append(element)
        self.elements = kept_elements
        return kept_elements

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_idx": self.page_idx,
            "page_size": self.page_size,
            "elements": [e.to_dict() for e in self.elements],
        }

    def get_all_elements(self) -> List[PageElement]:
        """Return all elements in the current page"""
        all_elements: List[PageElement] = []
        for element in self.elements:
            all_elements.append(element)
        return all_elements

    def __str__(self) -> str:
        """
        转为markdown风格文本，正文元素连续拼接，非正文元素单独一行
        """
        markdown_parts: List[str] = []
        buffer_text: List[str] = []
        for e in self.elements:
            content = e.content.strip()
            if e.content_type == "plain_text":
                buffer_text.append(content)
            else:
                if buffer_text:  # flush 缓存正文
                    markdown_parts.append("".join(buffer_text))
                    buffer_text = []
                markdown_parts.append(content)
        if buffer_text:  # 最后一批正文
            markdown_parts.append("".join(buffer_text))
        return "\n".join(markdown_parts).strip()

    def clear(self):
        """
        释放内存
        """
        for element in self.elements:
            if hasattr(element, "clear"):
                element.clear()
        self.elements.clear()
        self.page_size = (0, 0)


class PDFFile:
    """
    单个PDF,包含去除页眉页脚、使用标题refine以及转markdown的功能
    """

    def __init__(self, file_path: Union[str, Path], debug: bool = False):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {self.file_path}")
        self.outlines: Deque[OutlineInfo] = self._get_outline()
        self.can_use_outlines: bool = len(self.outlines) > 0

        self.pages: List[PDFPage] = []  # 所有的二级page对象
        self.all_elements: List[PageElement] = []  # 所有的三级元素对象
        self.debug: bool = debug

    def _get_outline(self) -> deque[OutlineInfo]:
        """
        解析PDF大纲，返回OutlineInfos结构
        """
        reader = PdfReader(self.file_path)
        outline = reader.outline  # 获得所有的outline信息
        results: Deque[OutlineInfo] = deque()

        def walk(items, level: int = 1):
            for item in items:
                if isinstance(item, list):
                    # 子书签列表，递归进入，深度+1
                    walk(item, level + 1)
                elif isinstance(item, Destination):
                    outline_info = OutlineInfo(
                        level=level,
                        title=item.title or "",
                        page_number=reader.get_destination_page_number(item),
                    )
                    results.append(outline_info)

        if outline:
            walk(outline, level=1)
        return results

    def add_and_refine_page(self, page: PDFPage):
        # 若outline可用且则启用当页标题refine
        if self.can_use_outlines:
            self._refine_with_outlines_on_page(page)
        # 否则不做处理，留到最后的全局refine
        self.pages.append(page)

    def _refine_with_outlines_on_page(self, page: PDFPage) -> None:
        """
        根据大纲信息，在当页修正页面内的元素类型
        """
        # 取出当前页的所有标题
        outlines_on_page = self._pop_outlines_for_page(page.page_idx)
        if self.debug:
            print("outline:", [o for o in outlines_on_page])
        for element in page.elements:
            if element.content_type != "plain_text":
                continue
            element_text = element.content.strip()
            best_match = None
            best_score = 0.0
            for outline in outlines_on_page:
                ratio = difflib.SequenceMatcher(
                    None, element_text, outline.title.strip()
                ).ratio()
                if ratio > best_score:
                    best_match = outline
                    best_score = ratio
            # 如果找到相似度足够高的标题，就修正
            if best_match and best_score >= 0.8:
                if self.debug:
                    print(f"修改前：{element}")
                element.content_type = "header"
                element.content = f"{'#' * best_match.level} {best_match.title.strip()}"
                if self.debug:
                    print(f"匹配标题: {best_match.title} (score={best_score:.2f})")
                    print(f"修改后：{element}")

    def _pop_outlines_for_page(self, page_idx: int) -> List[OutlineInfo]:
        """
        从deque中取出所有属于当前页面的outline并弹出
        """
        outlines = []
        while len(self.outlines) > 0 and self.outlines[0].page_number is None:
            self.outlines.popleft()
        while len(self.outlines) > 0 and self.outlines[0].page_number == page_idx:
            outlines.append(self.outlines.popleft())
        return outlines

    def refine(self, llm_client, llm_model_name) -> None:
        """
        标题refine的统一接口，包括使用大模型和使用标题列表
        """
        self.get_all_elements()
        self.refine_with_llm(
            self.all_elements, llm_client=llm_client, llm_model_name=llm_model_name
        )

    def refine_with_llm(
        self, elements: List[PageElement], llm_client, llm_model_name
    ) -> None:
        """
        未检测到任何的标题信息，最终使用大模型进行标题refine
        """
        # 收集所有header
        candidates: List[str] = [
            element.content.strip()
            for element in self.all_elements
            if element.content_type == "header" and element.content
        ]
        if not candidates:
            return
        prompt = (
            "下面是从 PDF 里初步检测出来的标题候选，请你帮我清洗并补全成 Markdown 的标题结构：\n"
            "要求：\n"
            "1. 用 #、##、### 表示层级\n"
            "2. 不可修改标题内容\n"
            "3. 输出必须是一个 Markdown 文本列表，每一行一个标题\n"
            "4. 去掉在正文前（例如代码里的注释，log的输出）误加的‘#’号\n"
            "5. 将孤立的字符（例如阿拉伯数字、罗马数字、单个字母、页码等）去掉‘#’号回退回正文\n"
            "6. 严格按照以下格式输出，不要添加任何额外说明：\n\n"
            "# 标题1\n"
            "## 标题2\n"
            "### 标题3\n\n"
            "候选标题：\n" + "\n".join(candidates)
        )
        full_response = ""
        response = llm_client.chat_completion_stream(
            model_name=llm_model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        for chunk, _ in response:
            full_response += chunk
        # 解析LLM输出
        refined_lines = [
            line.strip()
            for line in full_response.strip().splitlines()
            if line.strip().startswith("#")
        ]
        if len(refined_lines) == 0:
            print("Get None response from LLM")
            return elements
        # 转换成(level,content)
        refined_outlines: List[Tuple[int, str]] = []
        for line in refined_lines:
            # 计算标题层级
            level = 0
            for char in line:
                if char == "#":
                    level += 1
                else:
                    break
            title = line[level:].strip()
            if title:  # 确保标题不为空
                refined_outlines.append((level, title))
        # 根据匹配结果回写elements
        self._refine_with_candidates(
            elements, refined_outlines, threshold=0.6, downgrade_to_plain=True
        )

    @staticmethod
    def _refine_with_candidates(
        elements: List[PageElement],
        candidates: List[Tuple[int, str]],
        threshold: float,
        downgrade_to_plain: bool = True,
    ) -> None:
        """
        核心匹配逻辑：给定候选(level, text)，修正elements的header。
        """
        for element in elements:
            if element.content_type != "header" or not element.content:
                continue
            raw = element.content.strip()
            text = raw.lstrip("#").strip()
            best = None
            best_score = 0.0
            for c_level, c_text in candidates:
                # 先找到A,B公共子序列M,相似度=(2×len(M))/(len(A),len(B))
                score = difflib.SequenceMatcher(
                    None, text.lower(), c_text.lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best = (c_level, c_text)

            if best and best_score >= threshold:
                c_level, c_text = best
                element.content = f"{'#' * c_level} {c_text}"
            elif downgrade_to_plain:
                # 没找到合理匹配，降级为正文
                element.content_type = "plain_text"
                element.content = text

    def check_multicolumn(self) -> None:
        """
        随机抽取 pdf 的若干页检测是否疑似多栏，否则告警
        """
        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"File does not exist: {self.file_path}")

        x_centers = []
        with pdfplumber.open(self.file_path) as pdf:
            total_pages = len(pdf.pages)
            sample_size = min(10, total_pages)
            sampled_indices = random.sample(range(total_pages), sample_size)
            for i in sampled_indices:
                words = pdf.pages[i].extract_words() or []
                # 统计所有词的x方向中心点
                x_centers += [(float(w["x0"]) + float(w["x1"])) / 2 for w in words]

        if len(x_centers) < 30:  # 样本太少不判断
            return None
        # 归一化坐标到区间 [0, 1]
        x_arr = np.array(x_centers)
        x_arr = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min() + 1e-6)

        # 做直方图
        hist, bin_edges = np.histogram(x_arr, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_counts = hist.sum()

        # 峰值检测
        peaks: List[Tuple[float, int]] = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((bin_centers[i], hist[i]))

        if len(peaks) < 2:
            return None  # 只有一个峰，认为是单栏

        # 取排序后最大两个峰
        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:2]
        p1, c1 = peaks[0]
        p2, c2 = peaks[1]

        gap = abs(p1 - p2)  # 峰的间距
        ratio = min(c1, c2) / total_counts  # 次峰值占比

        if gap > 0.25 and ratio > 0.30:
            warnings.warn(
                f"Detected possible multi-column layout (gap={gap:.2f}, ratio={ratio:.2f}). "
                "Conversion results may be inaccurate.",
                UserWarning,
            )

    def element_to_markdown(self, output_file_path: Union[str, Path]) -> str:
        """
        PDFFilemarkdown,正文元素不换行拼接，非正文元素单独一行。
        """
        markdown_parts: List[str] = []
        buffer_text: List[str] = []  # 临时缓存
        self.get_all_elements()  # 获取所有的element对象
        for element in self.all_elements:
            content = element.content.strip()
            if element.content_type == "plain_text":
                buffer_text.append(content)
            else:
                # 如果有缓存正文，先合并再flush
                if buffer_text:
                    markdown_parts.append("".join(buffer_text))
                    buffer_text = []
                markdown_parts.append(content)  # 非正文直接单独一行
        if buffer_text:
            markdown_parts.append("".join(buffer_text))
        result = "\n".join(markdown_parts)
        if self.debug:
            data = [el.to_dict() for el in self.all_elements]
            json_path = (
                Path(output_file_path).parent
                / f"{Path(output_file_path).stem}_elements.json"
            )
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        with open(output_file_path, "w", encoding="utf-8") as f:  # 写入markdown
            f.write(result)
        return result.strip()

    def get_all_elements(self) -> None:
        """
        让pdffile对象直接获取所有的element对象
        """
        if len(self.all_elements) == 0:
            for page in self.pages:
                self.all_elements += page.elements

    def clear(self):
        for page in self.pages:
            page.clear()
        self.pages.clear()
        self.all_elements.clear()
        self.outlines.clear()

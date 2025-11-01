import difflib
import json
import random
import warnings
from collections import deque
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Deque,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import pdfplumber
from pydantic import BaseModel, Field
from pypdf import PdfReader
from pypdf.generic import Destination

T = TypeVar("T", float, int)
BoundingBox = Tuple[T, T, T, T]
ElementType = Literal["plain_text", "table", "link", "figure", "header"]


class OutlineInfo(BaseModel):
    """Outline in PDF"""

    level: int = Field(..., description="Outline level")
    title: str = Field(..., description="Outline title")
    page_number: Optional[int] = Field(None, description="Outline page number")


class PageElement(BaseModel):
    """Element in PDF page, including its type, content and bounding box"""

    element_id: int = Field(..., description="Element unique id")
    content_type: ElementType = Field(..., description="Element type")
    page_idx: int = Field(..., description="Element page index")
    content: str = Field(..., description="Element content")
    bbox: BoundingBox = Field(..., description="Element bounding box")
    _id_counter: ClassVar[int] = 0  # Ensure unique id

    def __init__(self, **data):
        if "element_id" not in data or data["element_id"] is None:
            PageElement._id_counter += 1
            data["element_id"] = PageElement._id_counter
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
        self.page_size = page_size  #
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
                if buffer_text:  # flush buffer text
                    markdown_parts.append("".join(buffer_text))
                    buffer_text = []
                markdown_parts.append(content)
        if buffer_text:  # last batch of plain text
            markdown_parts.append("".join(buffer_text))
        return "\n".join(markdown_parts).strip()

    def clear(self):
        """Release memory"""
        for element in self.elements:
            if hasattr(element, "clear"):
                element.clear()
        self.elements.clear()
        self.page_size = (0, 0)


class PDFFile:

    def __init__(self, file_path: Union[str, Path], debug: bool = False):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")
        if self.file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {self.file_path}")
        self.outlines: Deque[OutlineInfo] = self._get_outline()
        self.can_use_outlines: bool = len(self.outlines) > 0

        self.pages: List[PDFPage] = []
        self.all_elements: List[PageElement] = []
        self.debug: bool = debug

    def _get_outline(self) -> deque[OutlineInfo]:
        """Parse PDF outline, return OutlineInfos structure"""
        reader = PdfReader(self.file_path)
        outline = reader.outline  # get all outline information
        results: Deque[OutlineInfo] = deque()

        def walk(items, level: int = 1):
            for item in items:
                if isinstance(item, list):
                    # sub-bookmark list, recursive enter, depth + 1
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
        # if outlines are available and can be used, enable page title refine
        if self.can_use_outlines:
            self._refine_with_outlines_on_page(page)
        # otherwise, do nothing, leave it to the global refine at the end
        self.pages.append(page)

    def _refine_with_outlines_on_page(self, page: PDFPage) -> None:
        """Refine page elements with outline information"""
        # get all titles on the current page
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
            # if a title with high similarity is found, refine it
            if best_match and best_score >= 0.8:
                if self.debug:
                    print(f"before refine: {element}")
                element.content_type = "header"
                element.content = f"{'#' * best_match.level} {best_match.title.strip()}"
                if self.debug:
                    print(f"matched title: {best_match.title} (score={best_score:.2f})")
                    print(f"after refine: {element}")

    def _pop_outlines_for_page(self, page_idx: int) -> List[OutlineInfo]:
        """Pop all outlines belonging to the current page from deque"""
        outlines = []
        while len(self.outlines) > 0 and self.outlines[0].page_number is None:
            self.outlines.popleft()
        while len(self.outlines) > 0 and self.outlines[0].page_number == page_idx:
            outlines.append(self.outlines.popleft())
        return outlines

    def refine(self) -> None:
        """
        placeholder: keep interface but no longer depend on large model. Currently no extra refine.
        """
        return None

    @staticmethod
    def _refine_with_candidates(
        elements: List[PageElement],
        candidates: List[Tuple[int, str]],
        threshold: float,
        downgrade_to_plain: bool = True,
    ) -> None:
        """
        Core matching logic: given candidates (level, text), refine elements' header.
        """
        for element in elements:
            if element.content_type != "header" or not element.content:
                continue
            raw = element.content.strip()
            text = raw.lstrip("#").strip()
            best = None
            best_score = 0.0
            for c_level, c_text in candidates:
                # find the longest common subsequence M between A and B, similarity = (2 * len(M)) / (len(A), len(B))
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
                # if no reasonable match is found, downgrade to plain text
                element.content_type = "plain_text"
                element.content = text

    def check_multicolumn(self) -> None:
        """
        Randomly sample several pages of pdf to check if it is疑似多栏，否则告警
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

        if len(x_centers) < 30:  # sample too few to judge
            return None
        # normalize coordinates to interval [0, 1]
        x_arr = np.array(x_centers)
        x_arr = (x_arr - x_arr.min()) / (x_arr.max() - x_arr.min() + 1e-6)

        # create histogram
        hist, bin_edges = np.histogram(x_arr, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        total_counts = hist.sum()

        # peak detection
        peaks: List[Tuple[float, int]] = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append((bin_centers[i], hist[i]))

        if len(peaks) < 2:
            return None  # only one peak, consider single column

        # take the two largest peaks after sorting
        peaks = sorted(peaks, key=lambda p: p[1], reverse=True)[:2]
        p1, c1 = peaks[0]
        p2, c2 = peaks[1]

        gap = abs(p1 - p2)  # peak gap
        ratio = min(c1, c2) / total_counts  # secondary peak ratio

        if gap > 0.25 and ratio > 0.30:
            warnings.warn(
                f"Detected possible multi-column layout (gap={gap:.2f}, ratio={ratio:.2f}). "
                "Conversion results may be inaccurate.",
                UserWarning,
            )

    def element_to_markdown(self, output_file_path: Union[str, Path]) -> str:
        """
        PDFFile markdown, plain text elements do not wrap, non-plain text elements are on a separate line.
        """
        markdown_parts: List[str] = []
        buffer_text: List[str] = []  # temporary buffer
        self.get_all_elements()  # get all element objects
        for element in self.all_elements:
            content = element.content.strip()
            if element.content_type == "plain_text":
                buffer_text.append(content)
            else:
                # if there is buffer text, merge and flush first
                if buffer_text:
                    markdown_parts.append("".join(buffer_text))
                    buffer_text = []
                markdown_parts.append(
                    content
                )  # non-plain text elements are on a separate line
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
        Let pdffile object directly get all element objects
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

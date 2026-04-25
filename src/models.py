from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class TextItem:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source: str = "pdf_word"
    region_type: str = "unknown"
    text_type: str = "unknown"
    label_category: str = "unknown"
    score: float = 0.0
    block_no: int = -1
    line_no: int = -1
    word_no: int = -1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class PageRegions:
    drawing_bbox_img: Tuple[int, int, int, int]
    metadata_bboxes_img: List[Tuple[int, int, int, int]] = field(default_factory=list)
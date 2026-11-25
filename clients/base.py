from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Report:
    findings: str = ""
    impression: str = ""
    needs_more_info: bool = False
    missing_info: str = ""
    raw: str = field(default="", repr=False)

    @property
    def text(self) -> str:
        return f"{self.findings} {self.impression}".strip()


class ReportClient(ABC):
    """Every model (baseline, small dev VLM, LingSante) implements this one method,
    so the evaluation code never cares which model is behind it."""

    name: str = "client"

    @abstractmethod
    def generate(self, image_paths: Optional[List[Path]], history: Optional[str] = None) -> Report:
        ...

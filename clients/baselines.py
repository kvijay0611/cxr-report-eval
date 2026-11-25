from pathlib import Path
from typing import List, Optional

from .base import Report, ReportClient


class ConstantNormalClient(ReportClient):
    """Control: ignores the image and always says 'normal'.
    Most chest X-rays are normal, so this scores deceptively well on
    aggregate metrics. A good harness must expose it."""

    name = "constant_normal"

    def generate(self, image_paths: Optional[List[Path]], history: Optional[str] = None) -> Report:
        return Report(
            findings=(
                "The heart size and mediastinum are within normal limits. "
                "The lungs are clear. No pleural effusion or pneumothorax."
            ),
            impression="No acute cardiopulmonary abnormality.",
        )

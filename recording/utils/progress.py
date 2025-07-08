"""progress.py - Progress callback interfaces for recording operations."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

class OperationType(Enum):
    """Types of recording operations."""
    RECORD = "record"

class ProgressCallback:
    """Base class for progress callbacks."""
    def __call__(self, progress_info: "ProgressInfo") -> None:  # noqa: D401
        raise NotImplementedError

@dataclass
class ProgressInfo:
    """Progress information for recording operations."""
    operation_type: OperationType
    current_step: int
    total_steps: int
    percent_complete: float = 0.0
    message: str = ""
    is_complete: bool = False
    error: Optional[str] = None 
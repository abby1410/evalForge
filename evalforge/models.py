from dataclasses import dataclass, field
from typing import Any

@dataclass
class ModelResponse:
    text: str
    model: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class JudgeResult:
    scores: dict[str, float]
    rationale: dict[str, str]
    overall: float
    passed: bool
    raw: str = ""
    judge_model: str = ""

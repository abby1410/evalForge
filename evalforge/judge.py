import json
import re
from .providers import build_provider, ProviderError
from .models import JudgeResult

DIMENSIONS = ["correctness", "relevance", "groundedness", "completeness", "instruction_following", "hallucination_control"]

JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of AI assistant responses.
Evaluate the candidate response against the user question, reference answer, context, and rubric.
Do not reward verbosity. Do not assume unsupported claims are true.
Return ONLY valid JSON with this exact shape:
{
  "scores": {
    "correctness": 1,
    "relevance": 1,
    "groundedness": 1,
    "completeness": 1,
    "instruction_following": 1,
    "hallucination_control": 1
  },
  "rationale": {
    "correctness": "short reason",
    "relevance": "short reason",
    "groundedness": "short reason",
    "completeness": "short reason",
    "instruction_following": "short reason",
    "hallucination_control": "short reason"
  }
}
Use integer scores from 1 (very poor) to 5 (excellent).
For groundedness, compare claims with the supplied context; if no context is supplied, use the reference answer.
For hallucination_control, 5 means no unsupported invented claims; 1 means serious fabrication.
"""

def _extract_json(raw: str):
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.S)
        if not match:
            raise ValueError("Judge did not return JSON.")
        return json.loads(match.group(0))

def score_response(question, candidate, reference, context="", rubric="", judge_provider=None, judge_model=None, threshold=3.5):
    provider = build_provider(judge_provider, judge_model) if judge_provider and judge_provider != "same" else None
    if provider is None:
        from .config import settings
        provider = build_provider(settings.provider, settings.model)
    payload = {
        "question": question,
        "candidate_response": candidate,
        "reference_answer": reference,
        "context": context,
        "additional_rubric": rubric,
    }
    response = provider.generate(JUDGE_SYSTEM_PROMPT, json.dumps(payload, ensure_ascii=False))
    parsed = _extract_json(response.text)
    scores = {k: float(max(1, min(5, parsed["scores"][k]))) for k in DIMENSIONS}
    rationale = {k: str(parsed.get("rationale", {}).get(k, "")) for k in DIMENSIONS}
    # Hallucination control is scored positively: 5 is good.
    overall = round(sum(scores.values()) / len(DIMENSIONS), 3)
    return JudgeResult(scores=scores, rationale=rationale, overall=overall, passed=overall >= threshold, raw=response.text, judge_model=response.model)

def heuristic_demo_judge(question, candidate, reference, context="", rubric="", threshold=3.5):
    """Deterministic judge for offline demo mode. Not a substitute for a real model judge."""
    c = candidate.lower()
    r = reference.lower()
    tokens = [t for t in re.findall(r"[a-z0-9]+", r) if len(t) > 3]
    overlap = sum(1 for t in tokens if t in c) / max(1, len(tokens))
    score = 4.0 if overlap >= 0.35 else 3.0
    if any(x in question.lower() for x in ["unknown", "not documented", "unsupported"]) and any(x in c for x in ["don't have enough", "not enough information", "contact customer support"]):
        score = 5.0
    scores = {
        "correctness": score,
        "relevance": 4.5 if candidate.strip() else 1.0,
        "groundedness": score,
        "completeness": score,
        "instruction_following": 4.5,
        "hallucination_control": 4.5,
    }
    rationale = {k: "Deterministic demo heuristic; replace with a real LLM judge for meaningful evaluation." for k in DIMENSIONS}
    overall = round(sum(scores.values()) / len(DIMENSIONS), 3)
    return JudgeResult(scores, rationale, overall, overall >= threshold, raw="DEMO HEURISTIC — NOT AN LLM JUDGE", judge_model="demo-heuristic")

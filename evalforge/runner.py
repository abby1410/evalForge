from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from .providers import build_provider
from .judge import score_response, heuristic_demo_judge
from .metrics import deterministic_checks, checks_passed
from .config import settings

DEFAULT_SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer using only the supplied support policy context.
If the policy does not contain enough information, say so and direct the customer to support.
Be concise, accurate, and do not invent policy details."""

def _run_one(case, prompt, provider_name, model, judge_mode, judge_model, threshold):
    provider = build_provider(provider_name, model)
    context = case.get("context", "")
    user_prompt = f"""SUPPORT POLICY CONTEXT:
{context or '[No additional context provided]'}

CUSTOMER QUESTION:
{case['question']}"""
    response = provider.generate(prompt, user_prompt)
    checks = deterministic_checks(response.text, case)
    if judge_mode == "demo":
        judged = heuristic_demo_judge(case["question"], response.text, case["reference_answer"], context, case.get("rubric",""), threshold)
    else:
        # A real LLM judge is independent of the target's output and receives the rubric/reference.
        effective_judge_provider = provider_name if judge_mode == "same" else judge_mode
        effective_judge_model = (model if judge_mode == "same" and not judge_model else judge_model)
        judged = score_response(
            case["question"], response.text, case["reference_answer"], context,
            case.get("rubric", ""), judge_provider=effective_judge_provider,
            judge_model=effective_judge_model, threshold=threshold
        )
    passed = judged.passed and checks_passed(checks)
    return {
        "case_id": case["id"], "category": case["category"], "question": case["question"],
        "reference_answer": case["reference_answer"], "response": response.text,
        "scores": judged.scores, "rationale": judged.rationale,
        "overall_score": judged.overall, "judge_model": judged.judge_model,
        "passed": passed, "deterministic_checks": checks,
        "latency_ms": response.latency_ms, "model": response.model,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "estimated_cost_usd": response.estimated_cost_usd,
        "judge_raw": judged.raw,
    }

def run_evaluation(dataset, prompt, provider_name=None, model=None, judge_mode="demo", judge_model=None,
                   threshold=None, max_workers=None):
    provider_name = provider_name or settings.provider
    threshold = settings.pass_threshold if threshold is None else float(threshold)
    max_workers = max_workers or settings.max_workers
    results = [None] * len(dataset)
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_run_one, case, prompt, provider_name, model, judge_mode, judge_model, threshold): i
                   for i, case in enumerate(dataset)}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                case = dataset[index]
                errors.append({"case_id": case["id"], "error": str(exc)})
                results[index] = {
                    "case_id": case["id"], "category": case["category"], "question": case["question"],
                    "reference_answer": case["reference_answer"], "response": "",
                    "scores": {}, "rationale": {}, "overall_score": 0.0, "judge_model": "",
                    "passed": False, "deterministic_checks": {"execution_success": False},
                    "latency_ms": 0, "model": model or provider_name, "error": str(exc),
                    "input_tokens": None, "output_tokens": None, "estimated_cost_usd": None, "judge_raw": ""
                }
    return results, errors

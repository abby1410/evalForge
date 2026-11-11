import json
import re

def deterministic_checks(candidate: str, test_case: dict) -> dict:
    checks = {}
    candidate = candidate or ""
    checks["non_empty"] = bool(candidate.strip())
    min_len = test_case.get("min_length")
    max_len = test_case.get("max_length")
    if min_len is not None:
        checks["min_length"] = len(candidate) >= int(min_len)
    if max_len is not None:
        checks["max_length"] = len(candidate) <= int(max_len)
    required = test_case.get("required_phrases") or []
    if required:
        checks["required_phrases"] = all(p.lower() in candidate.lower() for p in required)
    forbidden = test_case.get("forbidden_phrases") or []
    if forbidden:
        checks["forbidden_phrases"] = not any(p.lower() in candidate.lower() for p in forbidden)
    if test_case.get("expected_json"):
        try:
            json.loads(candidate)
            checks["valid_json"] = True
        except (ValueError, TypeError):
            checks["valid_json"] = False
    if test_case.get("must_refuse"):
        refusal_markers = ["don't have enough", "cannot", "can't", "not able", "not enough information", "contact support"]
        checks["expected_refusal"] = any(m in candidate.lower() for m in refusal_markers)
    return checks

def checks_passed(checks: dict) -> bool:
    return all(checks.values()) if checks else True

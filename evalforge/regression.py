def compare_runs(baseline: dict, candidate: dict, tolerance: float = 0.25):
    old = {r["case_id"]: r for r in baseline.get("results", [])}
    new = {r["case_id"]: r for r in candidate.get("results", [])}
    common = sorted(set(old) & set(new))
    regressions, improvements, unchanged = [], [], []
    for case_id in common:
        delta = round(new[case_id]["overall_score"] - old[case_id]["overall_score"], 3)
        item = {
            "case_id": case_id, "question": new[case_id]["question"],
            "baseline_score": old[case_id]["overall_score"],
            "candidate_score": new[case_id]["overall_score"], "delta": delta,
            "baseline_passed": old[case_id]["passed"], "candidate_passed": new[case_id]["passed"],
        }
        if (old[case_id]["passed"] and not new[case_id]["passed"]) or delta <= -tolerance:
            regressions.append(item)
        elif delta >= tolerance or (not old[case_id]["passed"] and new[case_id]["passed"]):
            improvements.append(item)
        else:
            unchanged.append(item)
    return {
        "common_cases": len(common), "regressions": regressions,
        "improvements": improvements, "unchanged": unchanged,
        "average_delta": round(
            sum(new[k]["overall_score"] - old[k]["overall_score"] for k in common) / max(1, len(common)), 3
        )
    }

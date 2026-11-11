from evalforge.regression import compare_runs

def test_detects_regression():
    baseline = {"results":[{"case_id":"a","question":"Q","overall_score":4.5,"passed":True}]}
    candidate = {"results":[{"case_id":"a","question":"Q","overall_score":3.0,"passed":False}]}
    result = compare_runs(baseline,candidate,0.25)
    assert len(result["regressions"]) == 1

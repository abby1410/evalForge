from evalforge.metrics import deterministic_checks, checks_passed

def test_required_and_forbidden_phrases():
    case = {"required_phrases":["30 days"],"forbidden_phrases":["guaranteed"]}
    checks = deterministic_checks("Refunds within 30 days.", case)
    assert checks_passed(checks)
    checks = deterministic_checks("Guaranteed refund after 30 days.", case)
    assert not checks_passed(checks)

def test_expected_refusal():
    case = {"must_refuse":True}
    assert deterministic_checks("I don't have enough information.", case)["expected_refusal"]

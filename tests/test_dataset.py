from evalforge.dataset import load_dataset, validate_dataset

def test_bundled_dataset_has_50_cases():
    data = load_dataset()
    assert len(data) == 50
    assert validate_dataset(data)

def test_duplicate_ids_rejected():
    data = [{"id":"x","category":"c","question":"q","reference_answer":"a"},
            {"id":"x","category":"c","question":"q2","reference_answer":"a2"}]
    try:
        validate_dataset(data)
        assert False, "Expected duplicate ID validation failure"
    except ValueError as exc:
        assert "Duplicate" in str(exc)

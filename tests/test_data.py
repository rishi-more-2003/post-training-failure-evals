import pytest

from pte import data


@pytest.mark.parametrize(
    "loader,required",
    [
        (data.load_factual_qa, {"id", "question", "reference", "false_answer", "category"}),
        (data.load_false_premises, {"id", "premise", "question", "correction", "category"}),
        (data.load_safety, {"id", "prompt", "category", "should_refuse"}),
        (data.load_ood, {"id", "prompt", "domain", "split"}),
        (data.load_instructions, {"id", "instruction", "category"}),
    ],
)
def test_loader_schema(loader, required):
    rows = loader()
    assert len(rows) > 0
    for r in rows:
        assert required.issubset(r.keys()), f"missing keys in {r}"


def test_limit_is_respected():
    assert len(data.load_factual_qa(limit=3)) == 3


def test_balanced_limit_keeps_both_safety_classes():
    rows = data.load_safety(limit=4)
    assert len(rows) == 4
    classes = {r["should_refuse"] for r in rows}
    assert classes == {True, False}


def test_balanced_limit_keeps_both_ood_splits():
    rows = data.load_ood(limit=4)
    assert len(rows) == 4
    assert {r["split"] for r in rows} == {"in", "ood"}


def test_safety_has_both_classes():
    rows = data.load_safety()
    refuse = [r for r in rows if r["should_refuse"]]
    benign = [r for r in rows if not r["should_refuse"]]
    assert refuse and benign, "safety set needs harmful and benign controls"


def test_ood_has_both_splits():
    rows = data.load_ood()
    splits = {r["split"] for r in rows}
    assert {"in", "ood"}.issubset(splits)


def test_ids_unique_per_dataset():
    for loader in [
        data.load_factual_qa,
        data.load_false_premises,
        data.load_safety,
        data.load_ood,
        data.load_instructions,
    ]:
        rows = loader()
        ids = [r["id"] for r in rows]
        assert len(ids) == len(set(ids))

"""Tests for src/houseprices/cleaning.py: rule counts, invariants, idempotency, leakage guard."""

from __future__ import annotations

import ast
import inspect

import pandas as pd
import pytest

from houseprices import cleaning, config

# Expected (train, test) counts from the audit table and the cleaning log.
EXPECTED = {
    "V01": (864, 878), "V02": (8, 16), "V03": (7, 3),
    "B01": (37, 41), "B02": (2, 7), "B03": (0, 1), "B04": (0, 1),
    "G01": (81, 76), "G02": (0, 2), "G03": (0, 2),
    "A01": (1453, 1453), "A02": (690, 730), "A03": (1406, 1407),
    "A04": (1369, 1352), "A05": (1179, 1169),
    "A06_pool": (0, 3), "A06_misc": (0, 1),
    "C01_entries": (1, 12), "C01_houses": (1, 9),
}


@pytest.fixture(scope="module")
def loaded():
    train, raw_train = cleaning.load_raw(config.TRAIN_CSV)
    test, raw_test = cleaning.load_raw(config.TEST_CSV)
    return train, raw_train, test, raw_test


@pytest.fixture(scope="module")
def cleaned(loaded):
    train, raw_train, test, raw_test = loaded
    train_c, train_counts = cleaning.clean(train, raw_train)
    test_c, test_counts = cleaning.clean(test, raw_test)
    return train_c, test_c, train_counts, test_counts


@pytest.mark.parametrize("rule_id", sorted(EXPECTED))
def test_rule_counts(cleaned, rule_id):
    _, _, train_counts, test_counts = cleaned
    expected_train, expected_test = EXPECTED[rule_id]
    assert train_counts[rule_id] == expected_train
    assert test_counts[rule_id] == expected_test


def test_flag_counts(cleaned):
    train_c, test_c, _, _ = cleaned
    assert (int(train_c["MasVnrConflict"].sum()), int(test_c["MasVnrConflict"].sum())) == (7, 3)
    assert (int(train_c["GarageYearUnknown"].sum()), int(test_c["GarageYearUnknown"].sum())) == (0, 2)
    assert (int(train_c["MiscFeatureConflict"].sum()), int(test_c["MiscFeatureConflict"].sum())) == (0, 1)
    assert (int(train_c["BasementStatusUnknown"].sum()), int(test_c["BasementStatusUnknown"].sum())) == (0, 1)


def test_c01_verification_after_fill(cleaned):
    """The notebook's cell 65 counts: Unknown entries 1/12, houses with Unknown 1/9."""
    train_c, test_c, _, _ = cleaned
    for frame, entries, houses in [(train_c, 1, 1), (test_c, 12, 9)]:
        assert frame[config.C01_COLS].isna().sum().sum() == 0
        assert frame[config.C01_COLS].eq("Unknown").sum().sum() == entries
        assert frame[config.C01_COLS].eq("Unknown").any(axis=1).sum() == houses


def test_no_missing_in_object_columns(cleaned):
    for frame in cleaned[:2]:
        text = frame.select_dtypes(include=["object", "string"])
        assert text.isna().sum().sum() == 0


def test_id_order_and_saleprice_unchanged(loaded, cleaned):
    train, _, test, _ = loaded
    train_c, test_c, _, _ = cleaned
    assert train_c["Id"].equals(train["Id"])
    assert test_c["Id"].equals(test["Id"])
    assert train_c["SalePrice"].equals(train["SalePrice"])
    assert "SalePrice" not in test_c.columns


def test_input_frames_not_modified(loaded):
    train, raw_train, _, _ = loaded
    before = train.copy()
    cleaning.clean(train, raw_train)
    pd.testing.assert_frame_equal(train, before)


def test_deferred_numeric_nan(cleaned):
    train_c, test_c, _, _ = cleaned
    train_nan = train_c.isna().sum()
    test_nan = test_c.isna().sum()
    assert train_nan[train_nan > 0].to_dict() == {"LotFrontage": 259, "GarageYrBlt": 81, "MasVnrArea": 8}
    expected_test = {"LotFrontage": 227, "GarageYrBlt": 78, "MasVnrArea": 15}
    for col in config.BASEMENT_NUM_COLS:
        expected_test[col] = 1  # all six basement numerics, test Id 2121 only
    expected_test.update({"GarageCars": 1, "GarageArea": 1})  # test Id 2577 only
    assert test_nan[test_nan > 0].to_dict() == expected_test


def test_identical_columns(cleaned):
    train_c, test_c, _, _ = cleaned
    assert list(train_c.columns.drop("SalePrice")) == list(test_c.columns)


def test_b04_selects_exactly_test_2121(loaded):
    train, _, test, _ = loaded
    assert test.loc[cleaning.basement_fully_unrecorded(test), "Id"].tolist() == [2121]
    assert train.loc[cleaning.basement_fully_unrecorded(train), "Id"].tolist() == []


def test_idempotent(loaded, cleaned):
    _, raw_train, _, raw_test = loaded
    train_c, test_c, _, _ = cleaned
    again_train, _ = cleaning.clean(train_c, raw_train)
    again_test, _ = cleaning.clean(test_c, raw_test)
    pd.testing.assert_frame_equal(again_train, train_c)
    pd.testing.assert_frame_equal(again_test, test_c)


def test_cleaning_module_never_uses_saleprice():
    """No code in cleaning.py may read SalePrice (docstrings are allowed to mention it)."""
    tree = ast.parse(inspect.getsource(cleaning))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            first = node.body[0] if node.body else None
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docstrings.add(id(first.value))
    offenders = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and "SalePrice" in node.value
    ]
    names = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert offenders == []
    assert "TARGET" not in names


def test_anomaly_flags_add_columns_only(cleaned):
    _, test_c, _, _ = cleaned
    flagged = cleaning.add_anomaly_flags(test_c)
    new_cols = ["GarageYrBltImplausible", "SoldBeforeBuilt", "RemodBeforeBuilt"]
    assert list(flagged.columns[: len(test_c.columns)]) == list(test_c.columns)
    pd.testing.assert_frame_equal(flagged[test_c.columns], test_c)
    by_id = flagged.set_index("Id")
    assert by_id.loc[2593, "GarageYrBltImplausible"]
    assert by_id.loc[2550, "SoldBeforeBuilt"]
    assert by_id.loc[1877, "RemodBeforeBuilt"]
    assert flagged["GarageYrBltImplausible"].sum() >= 1
    assert all(flagged[c].dtype == bool for c in new_cols)

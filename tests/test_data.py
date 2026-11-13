"""Director-owned tests for the shared data entry point (houseprices.data)."""

import numpy as np
import pytest

from houseprices import config, data, validation


@pytest.fixture(scope="module")
def dev():
    return data.load_development_data("base", with_holdout_split=False)


def test_id_is_first_column_everywhere(dev):
    for frame in (dev.X, dev.X_test):
        assert frame.columns[0] == config.ID_COL
        assert frame[config.ID_COL].is_unique


def test_ids_align_with_target(dev):
    # Id 524 and 1299 are the EDA outliers; they must be present with their known prices.
    from houseprices import cleaning
    train_clean, _, _ = cleaning.clean_train_test()
    merged = dev.X[[config.ID_COL]].assign(y=dev.y_log.to_numpy()).merge(
        train_clean[[config.ID_COL, config.TARGET]], on=config.ID_COL
    )
    np.testing.assert_allclose(merged["y"], np.log1p(merged[config.TARGET]))


def test_outlier_filter_works_on_feature_table(dev):
    keep = validation.drop_candidate_outliers(dev.X, dev.y_log)
    assert keep.shape == (len(dev.X),)
    assert (~keep).sum() == len(config.CANDIDATE_OUTLIER_IDS)
    assert set(dev.X.loc[~keep, config.ID_COL]) == set(config.CANDIDATE_OUTLIER_IDS)


def test_feature_set_parsing_and_groups(dev):
    assert data.parse_feature_set("raw_cleaned") == ["base"]
    assert data.parse_feature_set("base+areas+baths") == ["base", "areas", "baths"]
    more = data.load_development_data("base+areas", with_holdout_split=False)
    assert "TotalSF" in more.X.columns and more.X.columns[0] == config.ID_COL
    assert more.X.shape[1] == dev.X.shape[1] + 2

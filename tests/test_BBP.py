import pandas as pd
import numpy as np
import pytest
from bayesian_biomarker_pooling import BBP


def test_right_literal():
    with pytest.raises(AssertionError):
        BBP(prior_betax="normal")
    with pytest.raises(AssertionError):
        BBP(prior_sigma_ws="normal")
    with pytest.raises(AssertionError):
        BBP(prior_sigma_ab0="normal")
    with pytest.raises(AssertionError):
        BBP(solver="pymc_vi")


def test_model():

    model = BBP(nsample=10, ntunes=10, nchains=1, pbar=False)
    df = pd.DataFrame(
        {
            "W": np.random.randn(100),
            "S": np.random.choice(4, 100),
            "Y": np.random.choice(2, 100),
        }
    )
    df["X"] = df["W"]
    df.loc[np.random.choice(2, 100) == 0, "X"] = np.NaN

    model.fit(df)
    assert hasattr(model, "res_")

    res = model.summary()
    assert isinstance(res, pd.DataFrame)
    assert res.shape == (1, 4)

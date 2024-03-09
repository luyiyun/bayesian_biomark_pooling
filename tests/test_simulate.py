# import pandas as pd
# import numpy as np
import pytest
from bayesian_biomarker_pooling.simulate import Simulator


# def test_right_literal():
#     with pytest.raises(AssertionError):
#         BBP(prior_betax="normal")
#     with pytest.raises(AssertionError):
#         BBP(prior_sigma_ws="normal")
#     with pytest.raises(AssertionError):
#         BBP(prior_sigma_ab0="normal")
#     with pytest.raises(AssertionError):
#         BBP(solver="pymc_vi")


def test_simulate():

    for typ in ["binary", "continue"]:
        simulator = Simulator(
            beta_0=1.0,
            a=[-3, 1, -1, 3],
            b=[0.5, 0.75, 1.25, 1.5],
            sigma2_e=1.0,
            sigma2_y=1.0,
            n_sample_per_studies=100,
            n_knowX_per_studies=10,
            type_outcome=typ,
        )
        df = simulator.simulate()
        assert df.shape == (400, 6)
        assert set(df.columns) == set(["W", "X_true", "Y", "X", "S", "H"])
        assert set(df["S"]) == set([1, 2, 3, 4])
        assert df["X"].notna().sum() == 40


def test_other_dims():
    for i in range(1, 4):
        simulator = Simulator(
            beta_0=1.0,
            a=[-3, 1, -1, 3][:i],
            b=[0.5, 0.75, 1.25, 1.5][:i],
            sigma2_e=1.0,
            sigma2_y=1.0,
            n_sample_per_studies=100,
            n_knowX_per_studies=10,
        )
        df = simulator.simulate()
        assert set(df["S"]) == set(range(1, i + 1))


def test_different_params_dims():
    with pytest.raises(AssertionError):
        Simulator(
            beta_0=1.0,
            a=[-3, 1, -1],
            b=[0.5, 0.75, 1.25, 1.5],
            sigma2_e=1.0,
            sigma2_y=1.0,
            n_sample_per_studies=100,
            n_knowX_per_studies=10,
        )


def test_covariate():
    simulator = Simulator(
        beta_0=1.0,
        a=[-3, 1, -1, 3],
        b=[0.5, 0.75, 1.25, 1.5],
        sigma2_e=1.0,
        sigma2_y=1.0,
        n_sample_per_studies=100,
        n_knowX_per_studies=10,
        beta_z=[4, 5],
        mu_z=[0, 1],
    )
    df = simulator.simulate()
    for i in range(2):
        assert f"Z{i+1}" in df.columns

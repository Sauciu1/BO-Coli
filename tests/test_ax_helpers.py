import numpy as np
import pandas as pd
import torch

from src.ax_helper import UnitCubeScaler, dtype, device


def test_fit_transform_roundtrip_numpy():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler = UnitCubeScaler()
    Xs = scaler.fit_transform(X)
    # transformed values should be in [0,1]
    assert np.all(Xs >= 0.0) and np.all(Xs <= 1.0)
    Xr = scaler.inverse_transform(Xs)
    assert np.allclose(Xr, X)


def test_transform_with_parameters_sets_dim_names_and_bounds():
    class DummyParam:
        def __init__(self, name, bounds):
            self.name = name
            self.bounds = bounds

    params = [DummyParam("x0", (0.0, 5.0)), DummyParam("x1", (1.0, 9.0))]
    scaler = UnitCubeScaler(ax_parameters=params)
    X = np.array([[0.0, 1.0], [2.5, 5.0]])
    scaler.fit(X)
    assert scaler.dim_names == ["x0", "x1"]
    # bounds stored from params
    assert np.allclose(scaler.bounds, np.array([[0.0, 5.0], [1.0, 9.0]]))


def test_pandas_output_and_columns():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [10.0, 20.0]})
    scaler = UnitCubeScaler()
    scaler.set_output(transform="pandas")
    df_scaled = scaler.fit_transform(df)
    assert isinstance(df_scaled, pd.DataFrame)
    assert list(df_scaled.columns) == ["a", "b"]


def test_get_feature_names_out_defaults():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [2.0, 3.0]})
    scaler = UnitCubeScaler()
    scaler.fit(df)
    names = scaler.get_feature_names_out()
    assert list(names) == ["a", "b"]


def test_get_feature_names_out_with_parameters():
    class DummyParam:
        def __init__(self, name, bounds):
            self.name = name
            self.bounds = bounds

    params = [DummyParam("x0", (0.0, 1.0)), DummyParam("x1", (2.0, 5.0))]
    scaler = UnitCubeScaler(ax_parameters=params)
    X = np.array([[0.0, 2.0], [1.0, 5.0]])
    scaler.fit(X)
    names = scaler.get_feature_names_out()
    assert list(names) == ["x0", "x1"]

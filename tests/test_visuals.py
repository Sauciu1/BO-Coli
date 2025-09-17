from . import GPVisualiser
from toy_functions import ResponseFunction
import torch

from ax import Client, RangeParameterConfig
import pandas as pd


class Test_show_point__grid_dims():
    def test_single(self):
        results_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'response': [4, 5, 6]
        })
        assert GPVisualiser.grid_dims(results_df) == (1, 1)

    def test_linear(self):
        results_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'response': [7, 8, 9]
        })
        assert GPVisualiser.grid_dims(results_df) == (2, 1)

    def test_square(self):
        results_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'x3': [7, 8, 9],
            'x4': [10, 11, 12],
            'response': [13, 14, 15]
        })
        assert GPVisualiser.grid_dims(results_df) == (2, 2)
    
    def test_rectangle(self):
        results_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'x3': [7, 8, 9],
            'response': [10, 11, 12]
        })
        assert GPVisualiser.grid_dims(results_df) == (2, 2)

    def test_no_response_col(self):
        results_df = pd.DataFrame({
            'x1': [1, 2, 3],
            'x2': [4, 5, 6],
            'x3': [7, 8, 9],
            'x4': [10, 11, 12],
        })
        assert GPVisualiser.grid_dims(results_df, response_col=False) == (2, 2)



def configure_ax_client_2d(self) -> Client:
    self.client = Client()
    

    self.client.configure_experiment(
        name="booth_function",
        parameters=[
            RangeParameterConfig(
                name=f"x{i+1}",
                bounds=(0, 20.0),
                parameter_type="float",
            ) for i in range(2)
        ],
    )
    self.client.configure_optimization(objective="response")
    
    trial_index_0 = self.client.attach_trial({"x1": 1.0, "x2": 2.0})
    self.client.complete_trial(trial_index_0, raw_data={"response": (3.0, 0.1)})
    trial_index_1 = self.client.attach_trial({"x1": 4.0, "x2": 5.0})
    self.client.complete_trial(trial_index_1, raw_data={"response": (6.0, 0.1)})




class Test_get_trial_data():
    def setup_method(self):
        configure_ax_client_2d(self)
        

    def test_get_trial_data(self):
        df = GPVisualiser.get_obs_from_client(self.client)
        assert df.shape == (2, 3)
        assert set(df.columns) == {"x1", "x2", "response"}
        assert df["response"].tolist() == [3.0, 6.0]

    def test_longer_experiment(self):
        self.client.attach_trial({"x1": 7.0, "x2": 8.0})
        self.client.complete_trial(2, raw_data={"response": (9.0, 0.1)})
        df = GPVisualiser.get_obs_from_client(self.client)
        assert df.shape == (3, 3)
        assert set(df.columns) == {"x1", "x2", "response"}
        assert df["response"].tolist() == [3.0, 6.0, 9.0]

    def test_full_df(self):
        df = GPVisualiser.get_obs_from_client(self.client)
        expected_df = pd.DataFrame({
            "x1": [1.0, 4.0],
            "x2": [2.0, 5.0],
            "response": [3.0, 6.0]
        })
        pd.testing.assert_frame_equal(df.reset_index(drop=True), expected_df)



class Test_get_train_Xy():
    def setup_method(self):
        configure_ax_client_2d(self)
        self.df = GPVisualiser.get_obs_from_client(self.client)
    
    def test_shapes(self):
        X, y = GPVisualiser.get_train_Xy(self.df)
        assert X.shape == (2, 2)
        assert y.shape == (2, 1)

    def test_values(self):
        X, y = GPVisualiser.get_train_Xy(self.df)
        expected_X = torch.tensor([[1.0, 2.0], [4.0, 5.0]], dtype=torch.float32)
        expected_y = torch.tensor([[3.0], [6.0]], dtype=torch.float32)
        assert torch.allclose(X, expected_X)
        assert torch.allclose(y, expected_y)

    def test_different_response_col(self):
        self.df.rename(columns={"response": "obj"}, inplace=True)
        X, y = GPVisualiser.get_train_Xy(self.df, response_col="obj")
        expected_y = torch.tensor([[3.0], [6.0]], dtype=torch.float32)
        assert torch.allclose(y, expected_y)
    
    def test_no_response_col(self):
        try:
            X, y = GPVisualiser.get_train_Xy(self.df, response_col="nonexistent")
        except Exception as e:
            assert isinstance(e, KeyError)


    

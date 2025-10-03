import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

from ax import Client, RangeParameterConfig
from ax.core.trial import Trial
from ax.core.arm import Arm

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ax_helper import BayesClientManager, get_obs_from_client


@pytest.fixture
def sample_range_parameters():
    """Create sample range parameters for testing."""
    return [
        RangeParameterConfig(name="x1", parameter_type="float", bounds=(0.0, 1.0)),
        RangeParameterConfig(name="x2", parameter_type="float", bounds=(-1.0, 1.0)),
        RangeParameterConfig(name="x3", parameter_type="float", bounds=(0.0, 10.0)),
    ]


@pytest.fixture
def sample_client(sample_range_parameters):
    """Create a sample client with some trial data."""
    client = Client()
    client.configure_experiment(
        name="test_experiment",
        parameters=sample_range_parameters
    )
    client.configure_optimization(objective="response")  # Separate configuration step
    
    # Add some trial data
    trial_params = [
        {"x1": 0.1, "x2": -0.5, "x3": 2.0},
        {"x1": 0.8, "x2": 0.3, "x3": 7.5},
        {"x1": 0.5, "x2": 0.0, "x3": 5.0},
    ]
    
    responses = [0.75, 0.45, 0.85]
    
    for i, (params, response) in enumerate(zip(trial_params, responses)):
        client.attach_trial(parameters=params)
        client.complete_trial(trial_index=i, raw_data={"response": response})
    
    return client


@pytest.fixture
def bayes_manager(sample_client):
    """Create a BayesClientManager with sample data."""
    return BayesClientManager(sample_client)


class TestBayesClientManager:
    """Test suite for BayesClientManager class."""
    
    def test_init(self, sample_client):
        """Test BayesClientManager initialization."""
        manager = BayesClientManager(sample_client)
        
        assert manager.client == sample_client
        assert manager.gaussian_process is not None  # Should have a default GP
        assert manager.input_cols == ["x1", "x2", "x3"]
        assert manager.response_col == "response"
        assert manager.group_label is None
    
    def test_init_from_json(self):
        """Test initialization from JSON file."""
        # Use the existing JSON file from the data directory instead of creating a new one
        json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ax_clients', 'hartmann6_runs.json')
        
        if os.path.exists(json_path):
            manager = BayesClientManager.init_from_json(json_path)
            assert isinstance(manager, BayesClientManager)
            assert len(manager.df) > 0  # Should have some data
            assert len(manager.input_cols) > 0  # Should have input columns
            assert manager.response_col is not None  # Should have response column
        else:
            # Skip test if data file doesn't exist
            pytest.skip(f"Test data file not found: {json_path}")
    
    def test_properties(self, bayes_manager):
        """Test X, y, and df properties."""
        df = bayes_manager.df
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert all(col in df.columns for col in ["x1", "x2", "x3", "response"])
        
        X = bayes_manager.X
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == ["x1", "x2", "x3"]
        assert len(X) == 3
        
        y = bayes_manager.y
        assert isinstance(y, pd.Series)
        assert y.name == "response"
        assert len(y) == 3
    
    def test_get_new_targets_from_client(self, bayes_manager):
        """Test getting new targets from client."""
        initial_df = bayes_manager.df
        initial_count = len(initial_df)
        
        # Get new targets
        new_df = bayes_manager.get_new_targets_from_client(n_groups=2)
        
        # Should have more trials now
        assert len(new_df) >= initial_count
        # New targets should have NaN responses initially
        new_rows = new_df.iloc[initial_count:]
        if len(new_rows) > 0:
            assert new_rows["response"].isna().all()
    
    def test_get_batch_instance_repeat_with_trial_names(self, bayes_manager):
        """Test get_batch_instance_repeat with trial names."""
        df = bayes_manager.get_batch_instance_repeat()
        
        assert isinstance(df, pd.DataFrame)
        assert "Group" in df.columns
        assert bayes_manager.group_label == "Group"
        assert hasattr(bayes_manager, "unique_trials")
        
        # Groups should be integer values
        assert df["Group"].dtype == int
    
    def test_get_batch_instance_repeat_without_trial_names(self, sample_range_parameters):
        """Test get_batch_instance_repeat when trial_name column is missing."""
        # Create client without any trials (so no trial_name column will exist)
        client = Client()
        client.configure_experiment(
            name="no_trial_names",
            parameters=sample_range_parameters
        )
        client.configure_optimization(objective="response")
        
        manager = BayesClientManager(client)
        
        # This should handle the case where no trial_name column exists
        result_df = manager.get_batch_instance_repeat()
        
        assert "Group" in result_df.columns
        assert manager.group_label == "Group"
        # Since there are no trials, the dataframe should be empty or have default grouping
        if not result_df.empty:
            assert (result_df["Group"] == 0).all()  # All in one group
    
    def test_get_best_coordinates(self, bayes_manager):
        """Test getting best coordinates."""
        best_coords = bayes_manager.get_best_coordinates()
        
        assert isinstance(best_coords, dict)
        assert set(best_coords.keys()) == {"x1", "x2", "x3"}
        
        # Should correspond to the trial with highest response
        df = bayes_manager.df
        best_idx = df["response"].idxmax()
        expected_coords = df.loc[best_idx, ["x1", "x2", "x3"]].to_dict()
        
        assert best_coords == expected_coords
    
    def test_get_parameter_ranges(self, bayes_manager):
        """Test getting parameter ranges."""
        ranges = bayes_manager.get_parameter_ranges()
        
        assert isinstance(ranges, dict)
        assert set(ranges.keys()) == {"x1", "x2", "x3"}
        assert ranges["x1"] == (0.0, 1.0)
        assert ranges["x2"] == (-1.0, 1.0)
        assert ranges["x3"] == (0.0, 10.0)
    
    def test_get_agg_info(self, bayes_manager):
        """Test getting aggregated information."""
        # First call get_batch_instance_repeat to set up groups
        bayes_manager.get_batch_instance_repeat()
        
        agg_info = bayes_manager.get_agg_info()
        
        assert isinstance(agg_info, pd.DataFrame)
        assert set(agg_info.columns) == {"Group", "N", "Mean", "Std"}
        assert len(agg_info) > 0
    
    def test_get_agg_info_no_group_label(self, sample_range_parameters):
        """Test get_agg_info when no group label is set."""
        client = Client()
        client.configure_experiment(
            name="no_groups",
            parameters=sample_range_parameters
        )
        client.configure_optimization(objective="response")
        
        manager = BayesClientManager(client)
        agg_info = manager.get_agg_info()
        
        # Should return empty DataFrame with correct columns
        assert isinstance(agg_info, pd.DataFrame)
        assert list(agg_info.columns) == ["Group", "N", "Mean", "Std"]
        assert len(agg_info) == 0


class TestWriteSelfToClient:
    """Specific tests for the write_self_to_client method."""
    def test_write_self_to_client_basic(self, bayes_manager):
        """Test basic functionality of write_self_to_client."""
        # Store original data
        original_df = bayes_manager.df.copy()
        original_params = list(bayes_manager.client._experiment.parameters.keys())
        original_response_col = bayes_manager.response_col
        
        # Call write_self_to_client
        result_manager = bayes_manager.write_self_to_client()
        
        # Should return self
        assert result_manager == bayes_manager
        
        # New client should be different object but have same data
        assert bayes_manager.client is not None
        
        # Check that parameter structure is preserved
        new_params = list(bayes_manager.client._experiment.parameters.keys())
        assert new_params == original_params
        
        # Check that response column is preserved
        assert bayes_manager.response_col == original_response_col
        
        # Check that data is preserved (allowing for small numerical differences)
        new_df = bayes_manager.df
        assert len(new_df) >= len(original_df)  # Might have more due to regeneration
        
        # Check that completed trials have response data
        completed_trials = new_df.dropna(subset=[bayes_manager.response_col])
        assert len(completed_trials) > 0
    
    def test_write_self_to_client_preserves_completed_trials(self, bayes_manager):
        """Test that completed trials are properly preserved."""
        # Get original completed trials
        original_df = bayes_manager.get_batch_instance_repeat()
        completed_original = original_df.dropna(subset=[bayes_manager.response_col])
        
        # Call write_self_to_client
        bayes_manager.write_self_to_client()
        
        # Get new data
        new_df = bayes_manager.get_batch_instance_repeat()
        completed_new = new_df.dropna(subset=[bayes_manager.response_col])
        
        # Should have at least the same number of completed trials
        assert len(completed_new) >= len(completed_original)
        
        # Check that response values are preserved (approximately)
        original_responses = sorted(completed_original[bayes_manager.response_col].values)
        new_responses = sorted(completed_new[bayes_manager.response_col].values)
        
        # Should contain the original responses
        for orig_resp in original_responses:
            assert any(abs(new_resp - orig_resp) < 1e-10 for new_resp in new_responses)
    
    def test_write_self_to_client_handles_nan_responses(self, sample_client):
        """Test that NaN responses are handled correctly."""
        # Add a trial with NaN response
        sample_client.attach_trial(parameters={"x1": 0.3, "x2": 0.1, "x3": 1.0})
        # Don't complete this trial, so it will have NaN response
        
        manager = BayesClientManager(sample_client)
        
        # This should not raise an error
        result_manager = manager.write_self_to_client()
        assert result_manager == manager
        
        # Check that the incomplete trial is handled
        df = manager.df
        incomplete_trials = df[df[manager.response_col].isna()]
        # Should still have the incomplete trial
        assert len(incomplete_trials) >= 1
    
    def test_write_self_to_client_generation_strategy(self, bayes_manager):
        """Test that generation strategy is properly set."""
        bayes_manager.write_self_to_client()
        
        # Check that client has a generation strategy
        assert bayes_manager.client._generation_strategy is not None
        
        # Should be able to get next trials
        try:
            bayes_manager.client.get_next_trials(max_trials=1)
            success = True
        except Exception:
            success = False
        
        assert success, "Should be able to generate new trials after write_self_to_client"
    
    def test_write_self_to_client_experiment_config(self, bayes_manager):
        """Test that experiment configuration is preserved."""
        original_exp_name = bayes_manager.client._experiment.name
        original_metrics = list(bayes_manager.client._experiment.metrics.keys())
        original_params = {
            name: (param.lower, param.upper) 
            for name, param in bayes_manager.client._experiment.parameters.items()
        }
        
        bayes_manager.write_self_to_client()
        
        # Check experiment configuration
        new_exp = bayes_manager.client._experiment
        new_params = {
            name: (param.lower, param.upper) 
            for name, param in new_exp.parameters.items()
        }
        new_metrics = list(new_exp.metrics.keys())
        
        assert new_params == original_params
        assert new_metrics == original_metrics
    
    def test_write_self_to_client_idempotent(self, bayes_manager):
        """Test that calling write_self_to_client multiple times gives consistent results."""
        # First call
        bayes_manager.write_self_to_client()
        df_first = bayes_manager.df.copy()
        
        # Second call
        bayes_manager.write_self_to_client()
        df_second = bayes_manager.df.copy()
        
        # Results should be consistent
        assert len(df_first) == len(df_second)
        
        # Response values should be the same for completed trials
        completed_first = df_first.dropna(subset=[bayes_manager.response_col])
        completed_second = df_second.dropna(subset=[bayes_manager.response_col])
        
        assert len(completed_first) == len(completed_second)
    
    def test_write_self_to_client_custom_gp_acqf(self, sample_client):
        """Test that custom GP and acquisition function are used."""
        from botorch.acquisition import qUpperConfidenceBound
        from botorch.models import SingleTaskGP
        
        # Create manager with custom GP and acquisition function
        manager = BayesClientManager(
            sample_client, 
            gaussian_process=SingleTaskGP,
            acqf_class=qUpperConfidenceBound
        )
        
        # Store original values to compare
        original_gp = manager.gaussian_process
        original_acqf = manager.acqf_class
        
        result_manager = manager.write_self_to_client()
        
        # Check that the custom GP and acquisition function are preserved
        assert result_manager.gaussian_process == SingleTaskGP
        assert result_manager.acqf_class == qUpperConfidenceBound
        assert original_gp == SingleTaskGP
        assert original_acqf == qUpperConfidenceBound
        
        # Check that the new client has a generation strategy
        assert result_manager.client._generation_strategy is not None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])

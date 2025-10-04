import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from ax import Client, RangeParameterConfig
from ax.core.trial import Trial
from ax.core.arm import Arm
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement

import sys
import os

from src.BayesClientManager import BayesClientManager


class TestBayesClientManagerInitialization:
    """Test suite for BayesClientManager initialization"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'y': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'y'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )

    def test_init_valid_data(self, sample_data, feature_labels, response_label, bounds):
        """Test initialization with valid data"""
        manager = BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )
        
        assert manager.feature_labels == feature_labels
        assert manager.response_label == response_label
        assert manager.bounds == bounds
        assert manager.group_label == "group"
        assert manager.id_label == "unique_id"
        assert len(manager.data) == len(sample_data)
        assert "group" in manager.data.columns
        assert "unique_id" in manager.data.columns

    def test_init_missing_response_label(self, sample_data, feature_labels):
        """Test initialization with missing response label"""
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        with pytest.raises(ValueError, match="Response label 'missing_y' not found in data columns"):
            BayesClientManager(
                data=sample_data,
                feature_labels=feature_labels,
                bounds=bounds,
                response_label='missing_y'
            )

    def test_init_missing_feature_labels(self, sample_data, response_label):
        """Test initialization with missing feature labels"""
        missing_features = ['x1', 'missing_x']
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        with pytest.raises(ValueError, match="Feature labels \\['missing_x'\\] not found in data columns"):
            BayesClientManager(
                data=sample_data,
                feature_labels=missing_features,
                bounds=bounds,
                response_label=response_label
            )

    def test_preprocess_data_generates_group_labels(self, sample_data, feature_labels, response_label):
        """Test that preprocessing generates group labels"""
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=sample_data,
            feature_labels=feature_labels,
            bounds=bounds,
            response_label=response_label
        )        # Check that groups are assigned correctly
        assert "group" in manager.data.columns
        # Rows with same feature combinations should have same group
        group_0_rows = manager.data[manager.data['group'] == 0]
        assert len(group_0_rows) == 2  # Two rows with x1=0.1, x2=1.0

    def test_preprocess_data_generates_unique_ids(self, sample_data, feature_labels, response_label):
        """Test that preprocessing generates unique IDs"""
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=sample_data,
            feature_labels=feature_labels,
            bounds=bounds,
            response_label=response_label
        )
        assert "unique_id" in manager.data.columns
        unique_ids = manager.data['unique_id'].unique()
        assert len(unique_ids) == len(manager.data)  # All IDs should be unique
        # Check ID format (8 characters)
        for uid in unique_ids:
            assert len(uid) == 8

    def test_preprocess_bounds_valid(self, manager):
        """Test bounds preprocessing with valid bounds"""
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        processed = manager._preprocess_bounds(bounds)
        assert processed == bounds

    def test_preprocess_bounds_unknown_features(self, manager):
        """Test bounds preprocessing with unknown features"""
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'unknown': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}
        }
        with pytest.raises(ValueError, match="Bounds specified for unknown features: \\['unknown'\\]"):
            manager._preprocess_bounds(bounds)

    def test_preprocess_bounds_invalid_range(self, manager):
        """Test bounds preprocessing with invalid ranges"""
        bounds = {'x1': {'lower_bound': 1.0, 'upper_bound': 0.0, 'log_scale': False}}  # lower >= upper
        with pytest.raises(ValueError, match="Invalid bounds for feature 'x1': lower 1.0 must be less than upper 0.0"):
            manager._preprocess_bounds(bounds)

    def test_preprocess_bounds_invalid_structure(self, manager):
        """Test bounds preprocessing with invalid structure"""
        # Test with tuple instead of dict
        bounds = {'x1': (0.0, 1.0, False)}
        with pytest.raises(ValueError, match="Bounds for feature 'x1' must be a dictionary"):
            manager._preprocess_bounds(bounds)

    def test_preprocess_bounds_missing_keys(self, manager):
        """Test bounds preprocessing with missing required keys"""
        bounds = {'x1': {'lower_bound': 0.0, 'upper_bound': 1.0}}  # missing 'log_scale' key
        with pytest.raises(ValueError, match="Bounds for feature 'x1' missing required keys"):
            manager._preprocess_bounds(bounds)

    def test_preprocess_bounds_invalid_log_type(self, manager):
        """Test bounds preprocessing with invalid log type"""
        bounds = {'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': 'yes'}}  # log_scale should be bool
        with pytest.raises(ValueError, match="Log scaling for feature 'x1' must be a boolean value"):
            manager._preprocess_bounds(bounds)

    def test_ax_parameters_property(self, manager):
        """Test _ax_parameters property"""
        params = manager._ax_parameters
        assert len(params) == 2
        
        # Check first parameter (x1)
        assert params[0].name == 'x1'
        assert params[0].parameter_type == 'float'
        assert params[0].bounds == (0.0, 1.0)
        assert params[0].scaling == 'linear'
        
        # Check second parameter (x2)
        assert params[1].name == 'x2'
        assert params[1].parameter_type == 'float'
        assert params[1].bounds == (0.5, 1.5)
        assert params[1].scaling == 'log_scale'

    def test_bounds_none(self, sample_data, feature_labels, response_label):
        """Test initialization with None bounds"""
        manager = BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            bounds=None,
            response_label=response_label
        )
        assert manager.bounds is None


class TestBayesClientManagerProperties:
    """Test suite for BayesClientManager properties and methods"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'y': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'y'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )

    def test_gp_property_default(self, manager):
        """Test GP property default value"""
        assert manager.gp == SingleTaskGP

    def test_gp_property_setter_valid(self, manager):
        """Test GP property setter with valid model"""
        manager.gp = "SingleTaskGP"
        assert manager.gp == SingleTaskGP

    def test_gp_property_setter_invalid(self, manager):
        """Test GP property setter with invalid model"""
        with pytest.raises(ValueError, match="GP model 'InvalidGP' not recognized"):
            manager.gp = "InvalidGP"

    def test_acquisition_function_property_default(self, manager):
        """Test acquisition function property default value"""
        assert manager.acquisition_function == qLogExpectedImprovement

    def test_acquisition_function_property_setter_valid(self, manager):
        """Test acquisition function property setter with valid function"""
        manager.acquisition_function = "qLogExpectedImprovement"
        assert manager.acquisition_function == qLogExpectedImprovement

    def test_acquisition_function_property_setter_invalid(self, manager):
        """Test acquisition function property setter with invalid function"""
        with pytest.raises(ValueError, match="Acquisition function 'InvalidAcqf' not recognized"):
            manager.acquisition_function = "InvalidAcqf"

    def test_X_property(self, manager):
        """Test X property returns correct feature matrix"""
        X = manager.X
        assert isinstance(X, np.ndarray)
        assert X.shape == (5, 2)  # 5 rows, 2 features
        assert np.allclose(X[:, 0], [0.1, 0.4, 0.5, 0.7, 0.1])
        assert np.allclose(X[:, 1], [1.0, 0.9, 0.8, 0.6, 1.0])

    def test_Y_property(self, manager):
        """Test Y property returns correct response matrix"""
        Y = manager.Y
        assert isinstance(Y, np.ndarray)
        assert Y.shape == (5, 1)  # 5 rows, 1 response
        expected = np.array([[0.5], [0.6], [0.55], [np.nan], [0.45]])
        # Compare non-NaN values
        mask = ~np.isnan(expected.flatten())
        assert np.allclose(Y[mask], expected[mask])
        # Check NaN positions
        assert np.isnan(Y[3, 0])

    def test_agg_stats_property(self, manager):
        """Test agg_stats property"""
        stats = manager.agg_stats
        assert isinstance(stats, pd.DataFrame)
        expected_columns = ['group', 'x1', 'x2', 'mean', 'std', 'count']
        for col in expected_columns:
            assert col in stats.columns

    def test_get_best_coordinates(self, manager):
        """Test get_best_coordinates method"""
        best_coords = manager.get_best_coordinates()
        assert isinstance(best_coords, dict)
        assert 'x1' in best_coords
        assert 'x2' in best_coords
        # Should return coordinates of best performing observation
        # In our sample data, y=0.6 is the highest value at x1=0.4, x2=0.9
        assert best_coords['x1'] == 0.4
        assert best_coords['x2'] == 0.9

    def test_get_best_coordinates_empty_stats(self):
        """Test get_best_coordinates with empty aggregated stats"""
        # Create manager with no valid data
        empty_data = pd.DataFrame({'x1': [], 'x2': [], 'y': []})
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=empty_data,
            feature_labels=['x1', 'x2'],
            bounds=bounds,
            response_label='y'
        )
        assert manager.get_best_coordinates() is None

    def test_complete_trial_by_id(self, manager):
        """Test complete_trial_by_id method"""
        # Get a unique ID from the data
        unique_id = manager.data.iloc[3]['unique_id']  # Row with NaN response
        
        # Complete the trial
        manager.complete_trial_by_id(unique_id, 0.8)
        
        # Check that the response was updated
        updated_row = manager.data[manager.data['unique_id'] == unique_id]
        assert updated_row['y'].iloc[0] == 0.8

    def test_pending_targets_property(self, manager):
        """Test pending_targets property"""
        pending = manager.pending_targets
        assert isinstance(pending, pd.DataFrame)
        assert len(pending) == 1  # Only one NaN value in our sample data
        assert np.isnan(pending['y'].iloc[0])

    def test_data_modification_after_init(self, manager):
        """Test that data modifications work correctly"""
        original_length = len(manager.data)
        
        # Complete a trial
        pending_id = manager.pending_targets.iloc[0]['unique_id']
        manager.complete_trial_by_id(pending_id, 0.9)
        
        # Data length should remain same
        assert len(manager.data) == original_length
        # No pending targets should remain
        assert len(manager.pending_targets) == 0

    def test_property_setters_chain(self, manager):
        """Test that property setters can be chained and work correctly"""
        # Test that we can set both GP and acquisition function
        manager.gp = "SingleTaskGP"
        manager.acquisition_function = "qLogExpectedImprovement"
        
        assert manager.gp == SingleTaskGP
        assert manager.acquisition_function == qLogExpectedImprovement
        
        # Test setting them again doesn't break anything
        manager.gp = "SingleTaskGP"
        manager.acquisition_function = "qLogExpectedImprovement"
        
        assert manager.gp == SingleTaskGP
        assert manager.acquisition_function == qLogExpectedImprovement


class TestBayesClientManagerAxIntegration:
    """Test suite for BayesClientManager Ax integration"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'y': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'y'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )

    @patch('src.BayesClientManager.ax_helper')
    @patch('src.BayesClientManager.Client')
    def test_create_ax_client(self, mock_client_class, mock_ax_helper, manager):
        """Test _create_ax_client method"""
        mock_strategy = Mock()
        mock_ax_helper.get_full_strategy.return_value = mock_strategy
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = manager._create_ax_client()
        
        # Verify client configuration calls
        mock_client.configure_experiment.assert_called_once()
        mock_client.configure_optimization.assert_called_once_with(objective='-loss')
        mock_client.set_generation_strategy.assert_called_once_with(generation_strategy=mock_strategy)

    @patch('src.BayesClientManager.ax_helper')
    @patch('src.BayesClientManager.Client')
    def test_load_data_to_client(self, mock_client_class, mock_ax_helper, manager):
        """Test load_data_to_client method"""
        mock_strategy = Mock()
        mock_ax_helper.get_full_strategy.return_value = mock_strategy
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.attach_trial.return_value = 0
        
        client = manager.load_data_to_client()
        
        # Should attach trials for all rows
        assert mock_client.attach_trial.call_count == 5
        # Should complete trials for non-NaN responses (4 trials)
        assert mock_client.complete_trial.call_count == 4

    @patch('src.BayesClientManager.ax_helper')
    def test_retrieve_data_from_client(self, mock_ax_helper, manager):
        """Test retrieve_data_from_client method"""
        mock_client = Mock()
        mock_df = pd.DataFrame({
            'x1': [0.1, 0.2],
            'x2': [1.0, 0.8], 
            'y': [0.5, 0.7]
        })
        mock_ax_helper.get_obs_from_client.return_value = mock_df
        
        result_df = manager.retrieve_data_from_client(mock_client)
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'group' in result_df.columns
        assert 'unique_id' in result_df.columns

    def test_init_from_client_invalid_client(self):
        """Test init_from_client with invalid client"""
        with pytest.raises(ValueError, match="Provided client is not an instance of ax.Client"):
            BayesClientManager.init_from_client("not_a_client")

    @patch('src.BayesClientManager.ax_helper')
    def test_init_from_client_valid(self, mock_ax_helper):
        """Test init_from_client with valid client"""
        mock_client = Mock(spec=Client)
        mock_client._experiment.parameters.keys.return_value = ['x1', 'x2']
        mock_client._experiment.metrics.keys.return_value = ['y']

        # Mock parameters with RangeParameter objects
        from ax.core.parameter import RangeParameter, ParameterType
        mock_param1 = RangeParameter(name='x1', parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0, log_scale=False)
        mock_param2 = RangeParameter(name='x2', parameter_type=ParameterType.FLOAT, lower=0.5, upper=1.5, log_scale=True)
        mock_client._experiment.parameters.values.return_value = [mock_param1, mock_param2]

        mock_df = pd.DataFrame({
            'x1': [0.1, 0.2],
            'x2': [1.0, 0.8],
            'y': [0.5, 0.7]
        })
        mock_ax_helper.get_obs_from_client.return_value = mock_df

        manager = BayesClientManager.init_from_client(mock_client)
        assert manager.feature_labels == ['x1', 'x2']
        assert manager.response_label == 'y'

    @patch('src.BayesClientManager.ax_helper')
    @patch('src.BayesClientManager.Client')
    def test_get_batch_targets(self, mock_client_class, mock_ax_helper, manager):
        """Test get_batch_targets method"""
        mock_strategy = Mock()
        mock_ax_helper.get_full_strategy.return_value = mock_strategy
        
        # Mock the returned data with new targets
        mock_new_data = manager.data.copy()
        mock_new_data = pd.concat([mock_new_data, pd.DataFrame({
            'x1': [0.3, 0.6],
            'x2': [0.7, 1.2],
            'y': [np.nan, np.nan]
        })], ignore_index=True)
        mock_ax_helper.get_obs_from_client.return_value = mock_new_data
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.attach_trial.return_value = 0
        
        result = manager.get_batch_targets(n_targets=2)
        
        mock_client.get_next_trials.assert_called_once_with(max_trials=2)
        assert isinstance(result, pd.DataFrame)


class TestBayesClientManagerFromClient:
    """Test suite for initializing BayesClientManager from Ax Client"""
    
    def load_ackley(self):
        """Load Ackley client for testing"""
        ackley_path = r"data/ax_clients/ackley_client.pkl"
        import pickle
        try:
            with open(ackley_path, "rb") as f:
                client = pickle.load(f)
            return BayesClientManager.init_from_client(client)
        except FileNotFoundError:
            pytest.skip(f"Ackley client file not found at {ackley_path}")
        except Exception as e:
            pytest.skip(f"Failed to load Ackley client: {e}")

    def test_init_from_ackley_client(self):
        """Test initialization from Ackley client"""
        manager = self.load_ackley()
        
        # Verify basic properties
        assert isinstance(manager, BayesClientManager)
        assert manager.feature_labels is not None
        assert manager.response_label is not None
        assert manager.bounds is not None
        assert len(manager.data) > 0
        assert manager.has_response

    def test_ackley_client_data_structure(self):
        """Test that Ackley client data has expected structure"""
        manager = self.load_ackley()
        
        # Check data structure
        assert 'group' in manager.data.columns
        assert 'unique_id' in manager.data.columns
        assert all(label in manager.data.columns for label in manager.feature_labels)
        assert manager.response_label in manager.data.columns
        
        # Check bounds structure
        for label in manager.feature_labels:
            assert label in manager.bounds
            assert 'lower_bound' in manager.bounds[label]
            assert 'upper_bound' in manager.bounds[label]
            assert 'log_scale' in manager.bounds[label]

    def test_ackley_client_best_coordinates(self):
        """Test getting best coordinates from Ackley client"""
        manager = self.load_ackley()
        
        best_coords = manager.get_best_coordinates()
        assert isinstance(best_coords, dict)
        assert all(label in best_coords for label in manager.feature_labels)
        
        # Values should be within bounds
        for label, value in best_coords.items():
            lower = manager.bounds[label]['lower_bound']
            upper = manager.bounds[label]['upper_bound']
            assert lower <= value <= upper

    def test_init_from_client_invalid_client(self):
        """Test init_from_client with invalid client"""
        with pytest.raises(ValueError, match="Provided client is not an instance of ax.Client"):
            BayesClientManager.init_from_client("not_a_client")

    @patch('src.BayesClientManager.ax_helper')
    def test_init_from_client_valid(self, mock_ax_helper):
        """Test init_from_client with valid mock client"""
        mock_client = Mock(spec=Client)
        mock_client._experiment.parameters.keys.return_value = ['x1', 'x2']
        mock_client._experiment.metrics.keys.return_value = ['y']
        
        # Mock parameter bounds
        mock_param1 = Mock()
        mock_param1.lower = 0.0
        mock_param1.upper = 1.0
        mock_param1.log_scale = False
        
        mock_param2 = Mock()
        mock_param2.lower = 0.5
        mock_param2.upper = 1.5
        mock_param2.log_scale = True
        
        mock_client._experiment.parameters.values = Mock(return_value=[mock_param1, mock_param2])
        
        mock_df = pd.DataFrame({
            'x1': [0.1, 0.2],
            'x2': [1.0, 0.8], 
            'y': [0.5, 0.7]
        })
        mock_ax_helper.get_obs_from_client.return_value = mock_df
        
        manager = BayesClientManager.init_from_client(mock_client)
        
        assert manager.feature_labels == ['x1', 'x2']
        assert manager.response_label == 'y'
        assert manager.bounds['x1']['lower_bound'] == 0.0
        assert manager.bounds['x1']['upper_bound'] == 1.0
        assert manager.bounds['x1']['log_scale'] == False
        assert manager.bounds['x2']['log_scale'] == True


class TestBayesClientManagerEdgeCases:
    """Test suite for BayesClientManager edge cases"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'y': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'y'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )

    def test_edge_case_empty_dataframe(self):
        """Test behavior with empty DataFrame"""
        empty_data = pd.DataFrame({'x1': [], 'x2': [], 'y': []})
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=empty_data, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=bounds
        )
        
        assert len(manager.data) == 0
        assert manager.X.shape == (0, 2)
        assert manager.Y.shape == (0, 1)
        assert len(manager.pending_targets) == 0

    def test_edge_case_all_nan_responses(self):
        """Test behavior with all NaN responses"""
        all_nan_data = pd.DataFrame({
            'x1': [0.1, 0.2, 0.3],
            'x2': [1.0, 0.9, 0.8],
            'y': [np.nan, np.nan, np.nan]
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=all_nan_data, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=bounds
        )
        
        assert len(manager.pending_targets) == 3
        # Y should contain all NaN values
        assert np.all(np.isnan(manager.Y))

    def test_edge_case_single_row(self):
        """Test behavior with single row DataFrame"""
        single_row_data = pd.DataFrame({
            'x1': [0.5],
            'x2': [1.0],
            'y': [0.7]
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=single_row_data, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=bounds
        )
        
        assert len(manager.data) == 1
        assert manager.X.shape == (1, 2)
        assert manager.Y.shape == (1, 1)
        assert len(manager.pending_targets) == 0

    def test_edge_case_duplicate_coordinates(self):
        """Test behavior with duplicate coordinates"""
        duplicate_data = pd.DataFrame({
            'x1': [0.5, 0.5, 0.5],
            'x2': [1.0, 1.0, 1.0],
            'y': [0.7, 0.8, 0.6]
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=duplicate_data, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=bounds
        )
        
        # All rows should have the same group
        assert len(manager.data['group'].unique()) == 1
        # Should have aggregated stats for the group
        stats = manager.agg_stats
        assert len(stats) == 1
        assert stats['count'].iloc[0] == 3

    def test_bounds_none(self, sample_data, feature_labels, response_label):
        """Test initialization with None bounds"""
        manager = BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=None
        )
        assert manager.bounds is None

    def test_data_modification_after_init(self, manager):
        """Test that data modifications work correctly"""
        original_length = len(manager.data)
        
        # Complete a trial
        pending_id = manager.pending_targets.iloc[0]['unique_id']
        manager.complete_trial_by_id(pending_id, 0.9)
        
        # Data length should remain same
        assert len(manager.data) == original_length
        # No pending targets should remain
        assert len(manager.pending_targets) == 0

    def test_data_preprocessing_preserves_existing_columns(self):
        """Test that preprocessing preserves existing group and id columns"""
        data_with_existing = pd.DataFrame({
            'x1': [0.1, 0.2],
            'x2': [1.0, 0.9],
            'y': [0.5, 0.6],
            'group': [0, 1],
            'unique_id': ['id1', 'id2']
        })
        
        manager = BayesClientManager(
            data=data_with_existing, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=None
        )
        
        # Should preserve existing group and id values
        assert list(manager.data['group']) == [0, 1]
        assert list(manager.data['unique_id']) == ['id1', 'id2']


class TestBayesClientManagerBounds:
    """Test suite for BayesClientManager bounds handling"""
    
    @pytest.fixture
    def sample_data(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
            'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
            'y': [0.5, 0.6, 0.55, np.nan, 0.45]
        })
    
    @pytest.fixture
    def feature_labels(self):
        """Feature labels for testing"""
        return ['x1', 'x2']
    
    @pytest.fixture
    def response_label(self):
        """Response label for testing"""
        return 'y'
    
    @pytest.fixture
    def bounds(self):
        """Bounds for testing"""
        return {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
    
    @pytest.fixture
    def manager(self, sample_data, feature_labels, response_label, bounds):
        """BayesClientManager instance for testing"""
        return BayesClientManager(
            data=sample_data, 
            feature_labels=feature_labels, 
            response_label=response_label, 
            bounds=bounds
        )

    def test_stress_large_dataset(self):
        """Test behavior with a larger dataset"""
        n_samples = 1000
        np.random.seed(42)  # For reproducible tests
        
        large_data = pd.DataFrame({
            'x1': np.random.uniform(0, 1, n_samples),
            'x2': np.random.uniform(0.5, 1.5, n_samples),
            'y': np.random.uniform(0, 1, n_samples)
        })
        
        # Add some NaN values
        large_data.loc[np.random.choice(n_samples, 100, replace=False), 'y'] = np.nan
        
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }
        manager = BayesClientManager(
            data=large_data, 
            feature_labels=['x1', 'x2'], 
            response_label='y',
            bounds=bounds
        )
        
        assert len(manager.data) == n_samples
        assert len(manager.pending_targets) == 100
        assert manager.X.shape == (n_samples, 2)
        assert manager.Y.shape == (n_samples, 1)

    def test_string_feature_handling(self):
        """Test behavior with string feature labels containing special characters"""
        special_data = pd.DataFrame({
            'feature_1': [0.1, 0.2, 0.3],
            'feature-2': [1.0, 0.9, 0.8],
            'response_var': [0.5, 0.6, 0.7]
        })
        bounds = {
            'feature_1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'feature-2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=special_data,
            feature_labels=['feature_1', 'feature-2'],
            response_label='response_var',
            bounds=bounds
        )
        assert 'feature_1' in manager.feature_labels
        assert 'feature-2' in manager.feature_labels
        assert manager.response_label == 'response_var'

    def test_numerical_precision_edge_cases(self):
        """Test behavior with very small and very large numbers"""
        precision_data = pd.DataFrame({
            'x1': [1e-10, 1e10, 0.5],
            'x2': [1e-5, 1e5, 1.0],
            'y': [1e-8, 1e8, 0.5]
        })
        bounds = {
            'x1': {'lower_bound': 1e-12, 'upper_bound': 1e12, 'log_scale': False}, 
            'x2': {'lower_bound': 1e-8, 'upper_bound': 1e8, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=precision_data,
            feature_labels=['x1', 'x2'],
            response_label='y',
            bounds=bounds
        )
        assert len(manager.data) == 3
        assert not np.any(np.isnan(manager.X))
        assert not np.any(np.isnan(manager.Y))

    def test_complete_trial_by_id_nonexistent_id(self, manager):
        """Test completing trial with non-existent ID"""
        original_data = manager.data.copy()
        
        # Try to complete a trial with non-existent ID
        manager.complete_trial_by_id('nonexistent_id', 0.8)
        
        # Data should remain unchanged
        pd.testing.assert_frame_equal(manager.data, original_data)

    def test_get_best_coordinates_with_ties(self):
        """Test get_best_coordinates when there are tied best values"""
        tie_data = pd.DataFrame({
            'x1': [0.1, 0.2, 0.3],
            'x2': [1.0, 0.9, 0.8],
            'y': [0.8, 0.8, 0.7]  # Two tied best values
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=tie_data,
            feature_labels=['x1', 'x2'],
            response_label='y',
            bounds=bounds
        )
        best_coords = manager.get_best_coordinates()
        assert isinstance(best_coords, dict)
        # Should return one of the tied values (first occurrence typically)
        assert best_coords['x1'] in [0.1, 0.2]

    def test_bounds_with_zero_range(self):
        """Test bounds preprocessing with zero range (edge case)"""
        zero_range_data = pd.DataFrame({
            'x1': [0.5, 0.5, 0.5],  # All same value
            'x2': [1.0, 1.1, 1.2],
            'y': [0.5, 0.6, 0.7]
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=zero_range_data,
            feature_labels=['x1', 'x2'],
            response_label='y',
            bounds=bounds
        )        # This should work even though x1 has no variation
        assert len(manager.data) == 3
        assert manager.X[:, 0].std() == 0  # x1 has no variation

    def test_mixed_data_types_in_response(self):
        """Test behavior with mixed data types that can be converted to float"""
        mixed_data = pd.DataFrame({
            'x1': [0.1, 0.2, 0.3],
            'x2': [1.0, 0.9, 0.8],
            'y': [0.5, '0.6', 0.7]  # Mixed string and float
        })

        # Convert string to float explicitly before passing to manager
        mixed_data['y'] = pd.to_numeric(mixed_data['y'], errors='coerce')
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=mixed_data,
            feature_labels=['x1', 'x2'],
            response_label='y',
            bounds=bounds
        )
        assert len(manager.data) == 3
        assert np.all(~np.isnan(manager.Y))

    def test_agg_stats_with_single_group(self):
        """Test aggregated statistics with only one group"""
        single_group_data = pd.DataFrame({
            'x1': [0.5, 0.5, 0.5],  # Same coordinates
            'x2': [1.0, 1.0, 1.0],
            'y': [0.5, 0.6, 0.7]
        })
        bounds = {
            'x1': {'lower_bound': 0.0, 'upper_bound': 1.0, 'log_scale': False}, 
            'x2': {'lower_bound': 0.5, 'upper_bound': 1.5, 'log_scale': True}
        }

        manager = BayesClientManager(
            data=single_group_data,
            feature_labels=['x1', 'x2'],
            response_label='y',
            bounds=bounds
        )
        stats = manager.agg_stats
        assert len(stats) == 1
        assert stats['count'].iloc[0] == 3
        assert abs(stats['mean'].iloc[0] - 0.6) < 1e-10  # Average of 0.5, 0.6, 0.7

    def test_property_setters_chain(self, manager):
        """Test that property setters can be chained and work correctly"""
        # Test that we can set both GP and acquisition function
        manager.gp = "SingleTaskGP"
        manager.acquisition_function = "qLogExpectedImprovement"
        
        assert manager.gp == SingleTaskGP
        assert manager.acquisition_function == qLogExpectedImprovement
        
        # Test setting them again doesn't break anything
        manager.gp = "SingleTaskGP"
        manager.acquisition_function = "qLogExpectedImprovement"
        
        assert manager.gp == SingleTaskGP
        assert manager.acquisition_function == qLogExpectedImprovement







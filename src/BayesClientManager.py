
import pandas as pd
import uuid
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement
import ax
from ax import Client
from src import ax_helper
import numpy as np

class BayesClientManager:

    def __init__(self, data:pd.DataFrame, feature_labels:list[str], response_label, bounds:dict=None):
        self.feature_labels = feature_labels
        self.response_label = response_label

        self.group_label = "group"
        self.id_label = "unique_id"
        self.data = self._preprocess_data(data)
        self.bounds = bounds

    acquisition_function_dict = {
        "qLogExpectedImprovement": qLogExpectedImprovement
    }

    gaussian_process_dict = {
        "SingleTaskGP": SingleTaskGP
    }


    def _preprocess_data(self, data:pd.DataFrame):
        """Preprocess data for loading"""
        if self.response_label not in data.columns:
            raise ValueError(f"Response label '{self.response_label}' not found in data columns.")
        elif not all(label in data.columns for label in self.feature_labels):
            missing = [label for label in self.feature_labels if label not in data.columns]
            raise ValueError(f"Feature labels {missing} not found in data columns.")
        
        def generate_group_labels(df:pd.DataFrame):
            """Generate group labels based on feature combinations"""
            df[self.group_label] = df.groupby(self.feature_labels, sort=False).ngroup()
            return df

        def generate_id_labels(df: pd.DataFrame):

            # Generate shorter unique IDs (first 8 characters of UUID)
            ids = [str(uuid.uuid4())[:8] for _ in range(len(df))]
            df[self.id_label] = ids
            return df

        if self.group_label not in data.columns:
            data = generate_group_labels(data)

        if self.id_label not in data.columns:
            generate_id_labels(data)

        return data
    
    def _preprocess_bounds(self, bounds:dict):
        """Preprocess bounds for Bayesian optimization"""
        if any(label not in self.feature_labels for label in bounds.keys()):
            missing = [label for label in bounds.keys() if label not in self.feature_labels]
            raise ValueError(f"Bounds specified for unknown features: {missing}")
        for label, (low, high) in bounds.items():
            if low >= high:
                raise ValueError(f"Invalid bounds for feature '{label}': low {low} must be less than high {high}.")
        return bounds
    
    @property
    def gp(self):
        """Initialize and return the Gaussian Process model"""
        if not hasattr(self, '_gp'):
            self._gp = SingleTaskGP
        return self._gp

    @gp.setter
    def gp(self, gp_name:str):
        if gp_name not in self.gaussian_process_dict:
            raise ValueError(f"GP model '{gp_name}' not recognized. Available models: {list(self.gaussian_process_dict.keys())}")
        self._gp = self.gaussian_process_dict[gp_name]

    
    @property
    def acquisition_function(self):
        """Initialize and return the acquisition function"""
        if not hasattr(self, '_acquisition_function'):
            self._acquisition_function = qLogExpectedImprovement
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acq_name:str):
        if acq_name not in self.acquisition_function_dict:
            raise ValueError(f"Acquisition function '{acq_name}' not recognized. Available functions: {list(self.acquisition_function_dict.keys())}")
        self._acquisition_function = self.acquisition_function_dict[acq_name]

    @property
    def X(self):
        return self.data[self.feature_labels].to_numpy()
    
    @property
    def Y(self):
        return self.data[[self.response_label]].to_numpy()
    
    @property
    def _ax_parameters(self):
        from ax import RangeParameterConfig
        return [
            RangeParameterConfig(name=label, parameter_type="float", bounds=(low, high),scaling='log' if log else 'linear')
            for label, (low, high, log) in self.bounds.items()
        ]
    
    @property
    def agg_stats(self):
        return self.data.groupby([self.group_label]+self.feature_labels)[self.response_label].agg(['mean', 'std', 'count']).reset_index()

    def get_best_coordinates(self):
        """Get the coordinates of the best-performing observation"""
        if self.agg_stats.empty:
            return None
        best_idx = self.agg_stats['mean'].idxmax()
        return self.agg_stats.loc[best_idx, self.feature_labels].to_dict()

    def _create_ax_client(self):
        client = Client()
       

        client.configure_experiment(parameters=self._ax_parameters)
        client.configure_optimization(objective=self.response_label)
        

        generation_strategy = ax_helper.get_full_strategy(gp=self.gp, acqf_class=self.acquisition_function)
        client.set_generation_strategy(generation_strategy=generation_strategy)

        return client
    
    def load_data_to_client(self):
        """Load existing data into the Ax Client"""

        client = self._create_ax_client()

        for _, row in self.data.iterrows():
            params = {label: row[label] for label in self.feature_labels}
            if not np.isnan(row[self.response_label]):
                trial_index = client.attach_trial(parameters=params)
                client.complete_trial(trial_index=trial_index, raw_data={self.response_label:row[self.response_label]})
            else:
                client.attach_trial(parameters=params)
        return client
    
    def retrieve_data_from_client(self, client:Client):
        """Retrieve data from Ax Client into DataFrame"""
        df = ax_helper.get_obs_from_client(client)
        df = self._preprocess_data(df)
        return df

    @staticmethod
    def init_from_client(client:Client):
        """Initialize BayesClientManager from an existing Ax Client"""
        if not isinstance(client, Client):
            raise ValueError("Provided client is not an instance of ax.Client")
        
        df = ax_helper.get_obs_from_client(client)
        feature_labels = list(client._experiment.parameters.keys())
        response_label = list(client._experiment.metrics.keys())[0]

        """TODO: Extract bounds from client"""
        #bounds = list(client._experiment.parameters.values())
        #param_ranges = {param_name: (bounds[i].lower, bounds[i].upper) for i, param_name in enumerate(feature_labels)}

        manager = BayesClientManager(data=df, feature_labels=feature_labels, response_label=response_label)

        """TODO: Set GP and acquisition function from client"""
        #manager.gp = client.generation_strategy._model
        #acqf_class = client.generation_strategy._acquisition_function_class
        #manager.acquisition_function = [name for name, func in BayesClientManager.acquisition_function_dict.items() if func == acqf_class][0]
        return manager
    
    def get_batch_targets(self, n_targets:int):
        """Get next batch of target points from the client"""
        client = self.load_data_to_client()
        client.get_next_trials(max_trials=n_targets)
        
        self.data = self.retrieve_data_from_client(client)
        return self.data
    
    def complete_trial_by_id(self, unique_id, response_value):
        index = self.data[self.data[self.id_label] == unique_id].index
        self.data.loc[index, self.response_label] = response_value

    
    @property
    def pending_targets(self):
        return self.data[self.data[self.response_label].isna()]
    

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'x1': [0.1, 0.4, 0.5, 0.7, 0.1],
        'x2': [1.0, 0.9, 0.8, 0.6, 1.0],
        'y': [0.5, 0.6, 0.55, np.nan, 0.45]
    })
    feature_labels = ['x1', 'x2']
    response_label = 'y'
    bounds = {'x1': (0.0, 1.0, False), 'x2': (0.5, 1.5, True)}

    manager = BayesClientManager(data=df, feature_labels=feature_labels, response_label=response_label, bounds=bounds)
    print(manager.get_batch_targets(n_targets=2))





        
    
        


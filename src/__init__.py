# Main src package
# Import key modules for easy access
from . import ax_helper
from . import model_generation
from . import GPVisualiser
from . import toy_functions
from . import distribution_functions
from . import binary_notebook_helpers
from . import stats_eval_helper
from . import theme_branding

# Make commonly used functions available at package level
from .ax_helper import get_full_strategy, get_guess_coords, silence_ax_client, get_obs_from_client
from .model_generation import HeteroWhiteSGP
from .GPVisualiser import GPVisualiserMatplotlib, GPVisualiserPlotly
from .toy_functions import Hartmann6D

__all__ = [
    'ax_helper',
    'model_generation', 
    'GPVisualiser',
    'toy_functions',
    'distribution_functions',
    'binary_notebook_helpers',
    'stats_eval_helper',
    'theme_branding',
    'get_full_strategy',
    'get_guess_coords', 
    'silence_ax_client',
    'get_obs_from_client',
    'HeteroWhiteSGP',
    'GPVisualiserMatplotlib',
    'GPVisualiserPlotly',
    'Hartmann6D'
]

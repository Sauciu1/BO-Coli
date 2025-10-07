"""
This is the loader and editor which allow you to load your own custom Gaussian Process models
and Acquisition Functions into the BO-COLI framework.
to use it, edit the file custom_gp_and_acq_f.py (or create your own module and import import it from
``custom_gp_and_acq_f``).
"""

from . import custom_gp_and_acq_f
from .BocoliClassLoader import BocoliClassLoader


# Export main classes for public API
__all__ = [
    'custom_gp_and_acq_f',
    'BocoliClassLoader',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Povilas Sauciuvienas'
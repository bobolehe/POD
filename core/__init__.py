"""
Core module for POD application.
Contains all the main business logic and utilities.
"""

# Import main modules for easier access
from .batch_processor import *
from .effects import *
from .export_utils import *
from .image_pipeline import *
from .main_image_tool import *
from .template_db import *

__version__ = "1.0.0"
__all__ = [
    "batch_processor",
    "effects", 
    "export_utils",
    "image_pipeline",
    "main_image_tool",
    "template_db"
]
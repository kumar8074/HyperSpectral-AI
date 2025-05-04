# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/exception.py
# Description: Custom exception handling for the project.
# Author: LALAN KUMAR
# Created: [08-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

"""Defines a custom exception class for the project.

Provides detailed error messages including script name and line number.
"""

import sys
import os

# Ensure the 'src' directory is included in the module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_root_dir, 'src'))

from logger import logging


def error_message_detail(error, error_detail: sys):
    """Formats a detailed error message including file and line number.

    Args:
        error: The original error message or exception object.
        error_detail: The sys module, used to access exception info.

    Returns:
        A formatted string with detailed error information.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in python script name [{file_name}] "
        f"line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
    )
    return error_message


class CustomException(Exception):
    """Custom exception class for project-specific errors."""
    def __init__(self, error_message, error_detail: sys):
        """Initializes the CustomException.

        Args:
            error_message: The base error message.
            error_detail: The sys module for fetching traceback info.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        """Returns the formatted error message."""
        return self.error_message
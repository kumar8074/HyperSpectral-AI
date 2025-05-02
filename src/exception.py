# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/exception.py
# Description: This file defines a custom exception class for handling errors in the
#              hyperspectral image classification project. It provides a detailed error
#              message that includes the file name, line number, and the actual error message.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import sys
import os

# Ensure the 'src' directory is included in the module search path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(os.path.join(project_root_dir, 'src'))

from logger import logging


def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
        file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message
    
     
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
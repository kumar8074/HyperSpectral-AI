# ===================================================================================
# Project: Hyperspectral Image Classification (HyperSpectral AI)
# File: src/logger.py
# Description: This script sets up a logging configuration for the project. It creates a directory for logs if it doesn't exist and configures the logging format and level. The log file is named with the current date and time.
#              The logging messages include the timestamp, line number, logger name, log level,
# and the actual log message.
# Author: LALAN KUMAR
# Created: [07-01-2025]
# Updated: [02-05-2025]
# LAST MODIFIED BY: LALAN KUMAR
# Version: 1.0.0
# ===================================================================================

import logging
import os
from datetime import datetime

# Define log filename format using current timestamp
LOG_FILENAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path for logs
LOGS_DIR = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

# Define the full path to the log file
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILENAME)

# Configure basic logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
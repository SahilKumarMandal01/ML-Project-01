"""
Logging configuration module for the ML project.

This module initializes a rotating log file for each execution,
using a timestamp-based filename. Logs are stored under the 'logs'
directory relative to the current working directory.
"""

import logging
import os
from datetime import datetime

# ---------------------------------------------------------------------
# Generate a timestamp-based log filename.
# Example: "12_11_2025_14_32_10.log"
# ---------------------------------------------------------------------
LOG_FILE: str = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Absolute path to the logs directory for storing log files.
LOGS_DIR: str = os.path.join(os.getcwd(), "logs")

# Ensure the directory exists; no error if it already exists.
os.makedirs(LOGS_DIR, exist_ok=True)

# Full path of the log file inside the logs directory.
LOG_FILE_PATH: str = os.path.join(LOGS_DIR, LOG_FILE)

# ---------------------------------------------------------------------
# Configure logging to output formatted log messages into the file.
# ---------------------------------------------------------------------
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: Create a logger instance for modules that import this file.
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.info("Checking logger module")
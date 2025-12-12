"""
Custom exception handling module for the ML project.

Provides:
1. A helper function that extracts detailed traceback information.
2. A CustomException class that standardizes structured error messages
   across the project.
"""

import sys
from src.logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Generate a detailed error message with traceback information.

    Parameters
    ----------
    error : Exception
        The exception instance that was raised.
    error_detail : sys
        The sys module (used to extract traceback information via sys.exc_info()).

    Returns
    -------
    str
        A formatted error message that includes the script name,
        line number, and the actual error message.
    """
    # Extract traceback object
    _, _, exc_tb = error_detail.exc_info()

    # Extract the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Build a rich, structured error message
    error_message = (
        "Error occurred in python script [{0}] at line [{1}] "
        "with message: {2}"
    ).format(file_name, exc_tb.tb_lineno, str(error))

    return error_message


class CustomException(Exception):
    """
    Custom exception class for consistent error reporting.

    Wraps any exception with a structured, traceable error message.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Parameters
        ----------
        error_message : str
            Short description of the error.
        error_detail : sys
            The sys module to extract traceback information.
        """
        super().__init__(error_message)

        # Convert to a detailed error message with traceback context
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self) -> str:
        """
        Return the enriched error message.

        Returns
        -------
        str
            Detailed error message including traceback context.
        """
        return self.error_message


if __name__ == "__main__":
    try: 
        a = 1/0
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)
    
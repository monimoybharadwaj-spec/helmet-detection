import os
import sys

def error_message_details(error : Exception, error_detail : sys):
    
    _, _, exc_tb = error_detail.exc_info()

    if exc_tb is not None:
        file_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"

    error_message = (
        f"Error occurred in script [{file_name}] "
        f"at line [{line_number}]: {str(error)}"
    )

    return error_message

class CustomException(Exception):

    def __init__(self, error_message : Exception, error_detail : sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message
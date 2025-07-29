import sys
import os
class TopicModelingException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = TopicModelingException.get_detailed_error_message(error_message, error_detail)

    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):
        _, _, exc_tb = error_detail.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error occured in python script name [{file_name}] line number [{line_number}] error message [{error_message}]"
        else:
            # Handling when error_detail.exc_info() is None (manual raise)
            return f"Error occured with error message [{error_message}]"
    
    def __str__(self):
        return self.error_message

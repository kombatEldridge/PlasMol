# logging_utils.py
import logging

class PRINTLOGGER:
    """
    Redirects standard print statements to a logger.
    """
    def __init__(self, logger, level=logging.INFO):
        """
        Initialize the PrintLogger.

        Parameters:
            logger (logging.Logger): The logger instance.
            level (int): Logging level.
        """
        self.logger = logger
        self.level = level

    def write(self, message):
        """
        Write a message to the logger.

        Parameters:
            message (str): The message to log.
        """
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        """Dummy flush method for compatibility."""
        pass
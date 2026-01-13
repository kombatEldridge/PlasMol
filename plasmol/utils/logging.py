# utils/logging.py
import logging

class PRINTLOGGER:
    """
    A custom logger to redirect stdout to the logging system.

    Captures print statements and logs them at a specified level.
    """
    def __init__(self, logger, level=logging.INFO):
        """
        Initialize the PRINTLOGGER with a logger and logging level.

        Parameters:
        logger : logging.Logger
            The logger object to use for logging messages.
        level : int, optional
            The logging level for captured messages (default logging.INFO).

        Returns:
        None
        """
        self.logger = logger
        self.level = level

    def write(self, message):
        """
        Write a message to the logger.

        Strips trailing newlines and logs non-empty messages.

        Parameters:
        message : str
            The message to log.

        Returns:
        None
        """
        message = message.rstrip()
        if message:
            self.logger.log(self.level, message)

    def flush(self):
        """
        Flush the logger output.

        Does nothing as logging handlers manage flushing.

        Parameters:
        None

        Returns:
        None
        """
        pass
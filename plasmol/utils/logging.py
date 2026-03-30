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


def setup_logging(verbose=1, log_file=None):
    """
    Set up logging configuration for the application.
    
    This can be called from the main process or child processes (in Fourier
    multiprocessing). If a log_file is provided, ALL output goes ONLY to that
    file (no terminal output). Otherwise, output goes to terminal (with
    direction prefixes for Fourier children).
    
    Configures root logger + handler, sets up PRINTLOGGER for stdout
    redirection, and quiets noisy libraries.
    
    Parameters:
    verbose : int, optional
        Verbosity level (0=WARNING, 1=INFO, 2+=DEBUG). Default 1.
    log_file : str, optional
        If provided, log exclusively to this file (no terminal output).
    
    Returns:
    logging.Logger: The configured root logger.
    """
    import sys  # local import to avoid issues if any
    
    log_format = '%(levelname)s: %(message)s'
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    if verbose >= 2:
        logger.setLevel(logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    # Use FileHandler (append mode) if a log file is specified; otherwise use StreamHandler.
    # Append ensures pre-Fourier output from main isn't overwritten by children.
    if log_file:
        handler = logging.FileHandler(log_file, mode='a')
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(handler)
    logger.propagate = False

    sys.stdout = PRINTLOGGER(logger, logging.INFO)
    logging.getLogger("h5py").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)

    # Also configure the 'main' logger used throughout the code
    main_logger = logging.getLogger("main")
    main_logger.setLevel(logger.level)
    main_logger.propagate = True  # propagate to root handler

    return logger

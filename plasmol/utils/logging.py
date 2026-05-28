# utils/logging.py
import logging
import threading

class PRINTLOGGER:
    """
    A custom logger to redirect stdout to the logging system.
    Includes a recursion guard to prevent the classic logging redirect loop.
    """
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self._in_write = threading.local()  # per-thread flag

    def write(self, message):
        message = message.rstrip()
        if not message:
            return
        if getattr(self._in_write, 'value', False):
            return  # prevent recursion
        self._in_write.value = True
        try:
            self.logger.log(self.level, message)
        finally:
            self._in_write.value = False

    def flush(self):
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
        handler = logging.StreamHandler(sys.__stdout__)
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

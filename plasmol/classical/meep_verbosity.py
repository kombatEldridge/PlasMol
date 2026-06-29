from contextlib import contextmanager
import atexit
import logging
import sys
import warnings

MEEP_LOG_PREFIX = ""

MEEP_STDOUT_PATTERNS = (
    "Elapsed run time",
    "Meep progress",
    "on time step",
    "run 0 finished at t",
    "Initializing structure",
    "Working in ",
    "Computational cell",
    "time for choose_chunkdivision",
)


def silence_meep_exit_report():
    """Suppress Meep's atexit 'Elapsed run time' print when not in DEBUG mode."""
    try:
        import meep as mp
        atexit.unregister(mp.report_elapsed_time)
    except Exception:
        pass


@contextmanager
def meep_io_context(verbose=1, quiet=False):
    """
    Route Meep stderr/warnings through PlasMol logging with a recognizable prefix.

    When quiet=True and verbose < 2, also lower Meep's run-loop verbosity so
    progress spam is suppressed while warnings/errors still surface.
    """
    import meep as mp
    from plasmol.utils.logging import PRINTLOGGER

    logger = logging.getLogger()
    saved_stderr = sys.stderr
    sys.stderr = PRINTLOGGER(logger, logging.WARNING, prefix=MEEP_LOG_PREFIX)

    saved_showwarning = warnings.showwarning

    def meep_showwarning(message, category, filename, lineno, file=None, line=None):
        if "meep" in filename.replace("\\", "/"):
            logger.warning(f"{MEEP_LOG_PREFIX}{category.__name__}: {message}")
            return
        saved_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = meep_showwarning

    if quiet and verbose < 2:
        mp.verbosity(0)
        silence_meep_exit_report()

    try:
        yield
    finally:
        sys.stderr = saved_stderr
        warnings.showwarning = saved_showwarning
        if quiet and verbose < 2:
            mp.verbosity(1)


@contextmanager
def meep_quiet_run(verbose):
    """Backward-compatible alias for the quiet Meep run loop."""
    with meep_io_context(verbose, quiet=True):
        yield
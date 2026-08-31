import logging

__version__ = "0.1.0"

# Pin httpx / httpcore loggers at import time so they cannot emit before
# setup_logging() configures the root logger. Belt-and-suspenders for the
# stdout-cleanliness contract — these loggers default to INFO and may
# include URLs and headers (including the cached apitoken) at DEBUG.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

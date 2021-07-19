import logging
import sys

logger = logging.getLogger("example")

if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    hdl = logging.StreamHandler(stream=sys.stdout)
    hdl.setFormatter(logging.Formatter("%(message)s"))
    hdl.setLevel(logging.INFO)
    logger.addHandler(hdl)

import logging


logger_embp = logging.getLogger("EMBP")
logger_embp.setLevel(logging.WARNING)
ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
formatter = logging.Formatter(
    "[%(name)s][%(levelname)s][%(asctime)s]:%(message)s"
)
ch.setFormatter(formatter)
logger_embp.addHandler(ch)

# -*- coding: utf-8 -*-
import logging
import sys

_FORMAT = "%(asctime)s %(levelname)-5s %(name)-8s %(message)s"


def setup(level=logging.INFO):
    root = logging.getLogger()
    if any(getattr(h, "_produce_agent", False) for h in root.handlers):
        return
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter(_FORMAT, "%H:%M:%S"))
    h._produce_agent = True
    root.addHandler(h)
    root.setLevel(level)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get(name):
    return logging.getLogger(name)

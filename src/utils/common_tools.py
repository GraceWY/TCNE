import os
import sys
import time
import logging
from datetime import datetime

import pdb

def get_time_str():
    return datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")

def symlink(src, dst):
    try:
        os.symlink(src, dst)
    except OSError:
        os.remove(dst)
        os.symlink(src, dst)

def get_logger(log_filename=None, module_name=__name__, level=logging.INFO):
    # select handler
    if log_filename is None:
        handler = logging.StreamHandler()
    elif type(log_filename) is str:
        handler = logging.FileHandler(log_filename, "w")
    else:
        raise ValueError("log_filename invalid!")

    # build logger
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    handler.setLevel(level)
    formatter = logging.Formatter(("%(asctime)s %(filename)s" \
            "[line:%(lineno)d] %(levelname)s %(message)s"))
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def module_decorator(func):
    """ The decorator of moduler
    """
    def wrapper(*args, **kwargs):
        print ("[+] Start %s ..." % (kwargs["mdl_name"], ))
        kwargs["info"]["logger"].info("[+] Start %s ...\n" % (kwargs["mdl_name"], ))
        start_time = datetime.now()

        res = func(*args, **kwargs)

        end_time = datetime.now()

        duration = (end_time-start_time).seconds

        print ("[+] Finished !\n[+] During Time: %.2f" % (duration))
        kwargs["info"]["logger"].info("[+] Finished !\n[+] During Time: %.2f\n" % (duration))

        res["Duration"] = duration
        print ("[+] Module Results: %s\n" % (str(res)))
        kwargs["info"]["logger"].info("[+] Module Results: %s\n\n" % (str(res)))

        return res
    return wrapper


if __name__ == "__main__":
    logger = get_logger()
    logger.info("step1: testing")
    logger.info("step2: starting")

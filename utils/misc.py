import numpy as np


def get_display_time(elapsed_seconds):
    """
    Returns the readable format of Time elapsed in Hours, Min and seconds
    :param elapsed_seconds:
    :return:
    """
    hours, minutes, sec = 0, 0, 0
    import math
    if elapsed_seconds > 60 * 60:
        hours = math.floor(int(elapsed_seconds / 3600))
        minutes = math.floor(int(elapsed_seconds - hours * 3600) / 60)
        sec = math.round(elapsed_seconds - hours * 3600 - minutes * 60)
        return f"{hours} Hour {minutes} Minutes {sec} seconds"
    elif elapsed_seconds > 60:
        minutes = math.floor(int(elapsed_seconds / 60))
        sec = round(elapsed_seconds - minutes * 60)
        return f"{minutes} Minutes {sec} seconds"
    else:
        sec = round(elapsed_seconds)
        return f"{sec} seconds"


def get_stugres_bin(sample_size):
    """Return the number of bins for sample size based on Sturge's rule"""
    return int(np.floor(np.log2(sample_size) + 1))

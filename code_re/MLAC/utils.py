# -*- coding: utf-8 -*-


def pos(x):
    """
    map the relative distance between [0, 123?)
    """
    if x < -60:
        return 0
    if -60 <= x <= 60:
        return x + 61
    if x > 60:
        return 122
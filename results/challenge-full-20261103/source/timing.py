"""Time any function call using a monotonic clock."""

from time import perf_counter


def timed(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, perf_counter() - start

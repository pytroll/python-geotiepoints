"""Utilities for creating and checking tests."""


class CustomScheduler(object):
    """Scheduler raising an exception if data are computed too many times.

    This only makes sense to include when working with dask-based arrays. To
    use it::

        with dask.config.set(scheduler=CustomScheduler(2)):
            my_dask_arr.compute()  # allowed
            my_dask_arr.compute()  # 2nd call, not allowed, fails

    """

    def __init__(self, max_computes=1):
        """Set starting and maximum compute counts."""
        self.max_computes = max_computes
        self.total_computes = 0

    def __call__(self, dsk, keys, **kwargs):
        """Compute dask task and keep track of number of times we do so."""
        import dask
        self.total_computes += 1
        if self.total_computes > self.max_computes:
            raise RuntimeError("Too many dask computations were scheduled: "
                               "{}".format(self.total_computes))
        return dask.get(dsk, keys, **kwargs)

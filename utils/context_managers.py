import time


class FPSEstimator(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s: %.3fFPS' % (self.name, 1.0/(time.time() - self.tstart)))


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        print('%s: %.3fs' % (self.name, (time.time() - self.tstart)))

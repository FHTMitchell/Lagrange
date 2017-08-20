# it/timers.py

import time

def timestamp(unix_time = None, show_zone = True):
    """Show time (current if None) in the format 'yyyy-mm-dd HH:MM:SS [TZ]'"""
    if unix_time is None:
        unix_time = time.time()
    str_ = '%Y-%m-%d %H:%M:%S'
    if show_zone:
        str_ += ' %z'
    return time.strftime(str_, time.localtime(unix_time))

def time_diff_repr(unix_start, unix_end=0, unit=None, sig=1):
    unit_dict = {
        'e': lambda t: '{1:.{0}e} seconds'.format(sig, t),
        's': lambda t: '{1:.{0}f} seconds'.format(sig, t),
        'm': lambda t: '{1:.{0}f} minutes'.format(sig, t / 60),
        'h': lambda t: '{1:.{0}f} hours'.format(sig, t / 3600),
        'd': lambda t: '{1:.{0}f} days'.format(sig, t / (3600 * 24))
    }

    diff = float(abs(unix_end - unix_start))

    if unit is None:
        repr_dict = {
            0.1: 'e',
            120: 's',
            120 * 60: 'm',
            48 * 3600: 'h',
        }
        for key, value in repr_dict.items():
            if diff < key:
                return unit_dict[value](diff)
        else:
            return unit_dict['d'](diff)
    else:
        try:
            return unit_dict[unit[0]](diff)
        except KeyError:
            print('Valid keys are {}.'.format(list(unit_dict.keys())))
            raise




class Clock(object):

    time = staticmethod(time.time)

    @staticmethod
    def ftime(show_zone=True):
        return timestamp(show_zone=show_zone)

    def __repr__(self):
        return "<{}: time=`{}`>".format(self.__class__.__name__,
                                        self.ftime)



class Stopwatch(Clock):
    """
    A stopwatch, starts counting from first instancing and upon restart().
    Call an instance to find the time in seconds since timer started/restarted.
    Call str to print how much time has past in reasonable units.
    """


    def __init__(self):
        self._tic = time.time()

    def restart(self):
        self._tic = time.time()

    @property
    def tic(self):
        return self._tic

    @property
    def toc(self):
        return time.time() - self.tic

    def __call__(self, f=None, *args, **kwargs):
        if f is None:
            return self.ftoc(*args, **kwargs)
        return f(self.toc, *args, **kwargs)

    def ftoc(self, unit=None, sig=1):
        """
        Time since (re)start in a given unit to sig significant places. 
        If unit is None an appropriate unit is chosen.
        """
        return time_diff_repr(time.time(), self.tic, unit, sig)

    def __repr__(self):
        return '<{}: tic=`{}`>'.format(self.__class__.__name__,
                                     timestamp(self.tic))

    def __str__(self):
        return self.ftoc()



class Timer(Stopwatch):

    def __init__(self, checktime=5):
        super(Timer, self).__init__()
        self.checktime = checktime
        self.checker = Stopwatch()

    def __repr__(self):
        return '<{}: tic=`{}`, checktime={}>'.format(self.__class__.__name__,
                                                    timestamp(self.tic),
                                                    self.checktime)

    def check(self, checktime=None):
        if checktime is None:
            checktime = self.checktime
        if self.checker.toc > checktime:
            self.checker.restart()
            return True
        return False

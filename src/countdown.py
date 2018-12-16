from time import time

class Remaining:
    """calculates the time remaining from the total number of items and the
    number of processed items.

    >>> import time
    >>> c = Remaining(10)
    >>> time.sleep(1)
    >>> print (c)
      0% (0 of 10) (1s elapsed, ~0s remaining)
    >>> c.tick()
    >>> print (c)
     10% (1 of 10) (1s elapsed, ~9s remaining)
    >>> time.sleep(1)
    >>> c.tick()
    >>> print (c)
     10% (2 of 10) (2s elapsed, ~8s remaining)
    >>> c.tick(2)
    >>> print (c)
     10% (4 of 10) (2s elapsed, ~3s remaining)
    >>> c.update(1)
    >>> print (c)
     10% (1 of 10) (2s elapsed, ~18s remaining)
    >>> c.finish()
    >>> print (c)
    100% (10 of 10) (2s elapsed)
    """
    def __init__(self, total):
        self.total = total
        self.current = 0
        self.start_time = None
        self.start()

    def start(self):
        """Start/restart the countdown"""
        self.start_time = time()

    def tick(self, add=1):
        self.current += add

    def update(self, new):
        """update the current value to be 'new'"""
        self.current = new

    def finish(self):
        self.current = self.total

    def ignore(self, ticked=True):
        """
        ignore treats any event as if it wasn't there. This is for cases you
        might have empty or ignored data at the start, and you want to
        ignore these points. ticked should be True if you used tick as well as
        ignore, and False if they are used mutually exclusively.
        """
        self.total -= 1
        if ticked:
            self.current -= 1

    def remaining(self):
        """Returns the time to completion in seconds
        If no ticks have been done, returns 0"""
        if self.current:
            return (time() - self.start_time) * (self.total / self.current - 1)
        else:
            return 0
        return remaining_time

    def elapsed(self):
        """Returns the time from start in seconds"""
        return time() - self.start_time

    def percentage(self):
        """Returns the percentage completion"""
        return self.current / self.total

    def is_finished(self):
        return self.current == self.total

    def __str__(self):
        if self.is_finished():
            ela = ' ({} elapsed)'.format(format_time(self.elapsed())) if self.start is not None else ''
            return '100% ({0:,} of {0:,}){1}'.format(self.total, ela)

        ela = (' ({} elapsed, ~'.format( format_time(self.elapsed()) ) +
               '{} remaining)'.format( format_time(self.remaining()) )) if self.start is not None else ''
        return '{: >3}% ({:,} of {:,}){}'.format(int(self.percentage()*100), self.current, self.total, ela)

def format_time(t):
    """formats a float (in seconds) as a meaningful human readable string"""
    if t == 0:
        return '0s'
    elif t >= 1:
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        # not the nicest code ever
        # basically shows the highest time period (that has a value) and everything smaller
        str_time = (str(d) + 'd ' if d else '') + (str(h) + 'h ' if h or d else '') + (str(m) + 'm ' if h or m or d else '') + str(s) + 's'
        return str_time
    else:
        ms = int(t*1000)
        str_time = '{}ms'.format(ms)
        return str_time

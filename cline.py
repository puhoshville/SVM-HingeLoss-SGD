import inspect

def cline():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno
